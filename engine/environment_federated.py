from __future__ import print_function
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from models.models import setup_model
from utils.utils import contains_class
from src.datasets import CustomDataset, distribute_dataset
import os
import sys
import random
from tqdm.std import tqdm
import copy
import time
from models.aggregation import average_weights, FLDefender
import gc
import torchvision
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from inversefed.reconstruction_algorithms import GradientReconstructor, DEFAULT_CONFIG
except ImportError:
    pass

try:
    with open(REPO_ROOT / "config.yaml", "r") as file:
        GLOBAL_CONFIG = yaml.safe_load(file)
except FileNotFoundError:
    print("Warning: 'config.yaml' not found. Using default paths.")
    GLOBAL_CONFIG = {}

INVERSION_CONFIG = GLOBAL_CONFIG.get("inversion", {})


BASE_DIR = REPO_ROOT / "model_checkpoints"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
SANITIZED_DIR = BASE_DIR / "sanitized_model"
RECON_DIR = REPO_ROOT / "reconstructed_images"


class Peer():
    _performed_attacks = 0
    # Small constants cached once on CPU; moved to device when needed.
    _BACKDOOR_PATTERN_CPU = {
        "MNIST": torch.tensor([[2.8238, 2.8238, 2.8238],
                               [2.8238, 2.8238, 2.8238],
                               [2.8238, 2.8238, 2.8238]], dtype=torch.float32),
        "CIFAR10": torch.tensor([[[2.5141, 2.5141, 2.5141], [2.5141, 2.5141, 2.5141], [2.5141, 2.5141, 2.5141]],
                                 [[2.5968, 2.5968, 2.5968], [2.5968, 2.5968, 2.5968], [2.5968, 2.5968, 2.5968]],
                                 [[2.7537, 2.7537, 2.7537], [2.7537, 2.7537, 2.7537], [2.7537, 2.7537, 2.7537]]], dtype=torch.float32),
    }
    _CIFAR_DM_CPU = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(3, 1, 1)
    _CIFAR_DS_CPU = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(3, 1, 1)
    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self,val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, local_data, labels, criterion, 
                device, local_epochs, local_bs, local_lr, 
                local_momentum, peer_type = 'honest'):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.peer_type = peer_type

#======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type = 'no_attack', malicious_behavior_rate = 0, 
                            source_class = None, target_class = None, dataset_name = None, global_rounds = None,
                            reconstruction_mode = False):
        
        # --- SETUP PATTERN ---
        if dataset_name not in self._BACKDOOR_PATTERN_CPU:
            raise ValueError(f"Unsupported dataset_name for backdoor pattern: {dataset_name}")
        backdoor_pattern = self._BACKDOOR_PATTERN_CPU[dataset_name].to(self.device)

        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        train_loader = DataLoader(self.local_data, self.local_bs, shuffle=True, drop_last=True, pin_memory=True, num_workers=2 if torch.cuda.is_available() else 0)
        
        if not reconstruction_mode:
            optimizer = optim.SGD(model.parameters(), lr=self.local_lr, momentum=self.local_momentum, weight_decay=5e-4)
            model.train()
        else:
            model.eval() 

        epochs_loss = []    

        for epoch in range(self.local_epochs):
            epoch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                is_attacking_now = False
                
                if (attack_type == 'backdoor') and (self.peer_type == 'attacker'):
                    if np.random.random() <= malicious_behavior_rate:
                        is_attacking_now = True
                        
                        pdata = data.clone(); ptarget = target.clone()
                        keep_idxs = (target == source_class)
                        pdata = pdata[keep_idxs]; ptarget = ptarget[keep_idxs]
                        pdata[:, :, -x_offset:, -y_offset:] = backdoor_pattern
                        ptarget[:] = target_class
                        
                        if reconstruction_mode:
                            if len(pdata) > 0:
                                target_img = pdata[-1].unsqueeze(0).detach().clone()
                                target_label = ptarget[-1].unsqueeze(0).detach().clone()
                            else:
                                is_attacking_now = False 
                        else:
                            data = torch.cat([data, pdata], dim=0)
                            target = torch.cat([target, ptarget], dim=0)

                if reconstruction_mode:
                    if is_attacking_now:
                        print(f"\n[RECON] Peer {self.peer_id} (Attacker) decided to attack in this round! Starting Gradient Inversion...")
                        try:
                            # Stats CIFAR10
                            dm = self._CIFAR_DM_CPU.to(self.device)
                            ds = self._CIFAR_DS_CPU.to(self.device)
                            
                            # Salva Originale
                            orig_show = torch.clamp(target_img[0] * ds + dm, 0, 1)
                            torchvision.utils.save_image(orig_show, f"{RECON_DIR}/ORIGINAL_peer{self.peer_id}.png")

                            # Calcola Gradiente
                            model.zero_grad()
                            loss = self.criterion(model(target_img), target_label)
                            input_gradient = torch.autograd.grad(loss, model.parameters())
                            input_gradient = [grad.detach().clone() for grad in input_gradient]

                            # Configurazione Inversion
                            config = DEFAULT_CONFIG.copy()

                            config['max_iterations'] = INVERSION_CONFIG.get('max_iterations', config['max_iterations'])
                            config['cost_fn'] = INVERSION_CONFIG.get('cost_fn', config['cost_fn'])
                            config['optim'] = INVERSION_CONFIG.get('optim', config['optim'])
                            config['lr'] = INVERSION_CONFIG.get('lr', config['lr'])
                            config['total_variation'] = INVERSION_CONFIG.get('total_variation', config['total_variation'])
                            config['boxed'] = INVERSION_CONFIG.get('boxed', config['boxed'])
                            config['restarts'] = INVERSION_CONFIG.get('restarts', config['restarts'])
                            config['lr_decay'] = INVERSION_CONFIG.get('lr_decay', config['lr_decay'])


                            rec_machine = GradientReconstructor(model, mean_std=(dm, ds), config=config, num_images=1)
                            onehot = torch.zeros(1, 10, device=self.device); onehot[0, target_class] = 1
                            
                            reconstructed, stats = rec_machine.reconstruct(input_gradient, onehot)

                            # Salva Risultato
                            rec_denorm = torch.clamp(reconstructed * ds + dm, 0, 1)
                            save_path = f"{RECON_DIR}/RECON_peer{self.peer_id}.png"
                            torchvision.utils.save_image(rec_denorm, save_path)
                            print(f"[RECON] Saved to {save_path} | Loss: {stats['opt']}")
                            
                        except Exception as e:
                            print(f"[RECON ERROR] {e}")
                        
                        # Reconstruction leaves parameter gradients around; clear them to avoid memory growth
                        # when running multiple attackers sequentially.
                        model.zero_grad(set_to_none=True)
                        return model, 0.0
                    else:
                        continue 

                model.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            
            if not reconstruction_mode:
                epochs_loss.append(np.mean(epoch_loss))
        
        model = model.cpu()
        return model, np.mean(epochs_loss) if epochs_loss else 0.0

class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_peers, frac_peers, 
    seed, test_batch_size, criterion, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, labels_dict, device, attackers_ratio = 0,
    class_per_peer=2, samples_per_class= 250, rate_unbalance = 1, alpha = 1,source_class = None):

        # Create output directories at runtime (not at module import time).
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(SANITIZED_DIR, exist_ok=True)
        os.makedirs(RECON_DIR, exist_ok=True)

        FL._history = np.zeros(num_peers)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.peers_pseudonyms = ['Peer ' + str(i+1) for i in range(self.num_peers)]
        self.frac_peers = frac_peers
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_peer = class_per_peer
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.peers = []
        self.trainset, self.testset = None, None
        self.score_history = np.zeros([self.num_peers], dtype = float)
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name, self.num_peers, self.num_classes, 
        self.dd_type, self.class_per_peer, self.samples_per_class, self.alpha)

        self.test_loader = DataLoader(self.testset, batch_size = self.test_batch_size,
            shuffle = False, num_workers = 1)
    
        self.global_model = setup_model(model_architecture = self.model_name, num_classes = self.num_classes, 
        tokenizer = tokenizer, embedding_dim = self.embedding_dim)
        self.global_model = self.global_model.to(self.device)
        
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        print('--> Distributing training data among peers')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            peer_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(peer_data)
            if  self.source_class in user_groups_train[p]['labels']:
                 self.have_source_class.append(p)
        print('--> Training data have been distributed among peers')

        print('--> Creating peers instances')
        m_ = 0
        if self.attackers_ratio > 0:
            k_src = len(self.have_source_class)
            print('# of peers who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * k_src)
            self.num_attackers = copy.deepcopy(m_)

        peers = list(np.arange(self.num_peers))  
        random.shuffle(peers)
        for i in peers:
            if m_ > 0 and contains_class(self.local_data[i], self.source_class):
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, peer_type = 'attacker'))
                m_-= 1
            else:
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum))  

        del self.local_data

#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name = None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            
            # NLP rimosso, manteniamo solo la logica CV
            test_loss.append(self.criterion(output, target).item()) 
            pred = output.argmax(dim=1, keepdim=True) 
            correct+= pred.eq(target.view_as(pred)).sum().item()

            n+= target.shape[0]
        test_loss = np.mean(test_loss)
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, n,
           100*correct / n))
        return  100.0*(float(correct) / n), test_loss

    def test_label_predictions(self, model, device, test_loader, dataset_name = None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                prediction = output.argmax(dim=1, keepdim=True)
                
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]
    
    def test_backdoor(self, model, device, test_loader, backdoor_pattern, source_class, target_class):
        model.eval()
        backdoor_pattern = backdoor_pattern.to(device)
        correct = 0
        n = 0
        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            keep_idxs = (target == source_class)
            bk_data = copy.deepcopy(data[keep_idxs])
            bk_target = copy.deepcopy(target[keep_idxs])
            bk_data[:, :, -x_offset:, -y_offset:] = backdoor_pattern
            bk_target[:] = target_class
            output = model(bk_data)
            pred = output.argmax(dim=1, keepdim=True) 
            correct+= pred.eq(bk_target.view_as(pred)).sum().item()
            n+= bk_target.shape[0]
        return  np.round(100.0*(float(correct) / n), 2)

    def choose_peers(self, use_reputation = False):
        m = max(int(self.frac_peers * self.num_peers), 1)
        selected_peers = np.random.choice(range(self.num_peers), m, replace=False)
        return selected_peers

    def update_score_history(self, scores, selected_peers, epoch):
        print('-> Update score history')
        self.score_history[selected_peers]+= scores
        q1 = np.quantile(self.score_history, 0.25)
        trust = self.score_history - q1 
        trust = trust/trust.max()
        trust[(trust < 0)] = 0
        return trust[selected_peers]
            
    def run_experiment(self, attack_type = 'no_attack', malicious_behavior_rate = 0,
        source_class = None, target_class = None, rule = 'fedavg', resume = False,
        reconstruction_only = False):
        
        simulation_model = copy.deepcopy(self.global_model)
        
        # Generazione dinamica del nome del checkpoint per la sessione corrente
        dynamic_checkpoint_name = f"{CHECKPOINT_DIR}/ckpt_{attack_type}_{self.dataset_name}_{self.model_name}_{rule}_{self.attackers_ratio}.t7"

        if reconstruction_only:
            # Se siamo in ricostruzione, prendiamo il modello avvelenato dal config
            recon_checkpoint = GLOBAL_CONFIG.get('unlearning', {}).get('poisoned_checkpoint', dynamic_checkpoint_name)
            # `config.yaml` uses relative paths; resolve them relative to repo root.
            recon_checkpoint_path = Path(str(recon_checkpoint))
            if not recon_checkpoint_path.is_absolute():
                recon_checkpoint = str((REPO_ROOT / recon_checkpoint_path).resolve())
            
            print("\n" + "="*60)
            print(f" MODALITÀ RICOSTRUZIONE ATTIVA (Skip Training)")
            print(f" Caricamento Checkpoint: {recon_checkpoint}")
            print("="*60)
            
            if not os.path.exists(recon_checkpoint):
                print(f"ERRORE: Il checkpoint {recon_checkpoint} non esiste. Esegui prima il training.")
                return

            checkpoint = torch.load(recon_checkpoint, map_location=self.device, weights_only=False)
            if 'state_dict' in checkpoint:
                simulation_model.load_state_dict(checkpoint['state_dict'])
            else:
                simulation_model.load_state_dict(checkpoint)
            
            simulation_model.to(self.device)
            
            print("--> Avvio simulazione round per intercettazione...")
            selected_peers = self.choose_peers()
            
            attackers_in_round = [p for p in selected_peers if self.peers[p].peer_type == 'attacker']
            
            if not attackers_in_round:
                print("--> Nessun attaccante selezionato in questo round. Riprova.")
                return

            for peer_idx in selected_peers:
                if self.peers[peer_idx].peer_type == 'attacker':
                    self.peers[peer_idx].participant_update(
                        global_epoch=999, 
                        model=simulation_model,
                        attack_type=attack_type,
                        malicious_behavior_rate=malicious_behavior_rate, 
                        source_class=source_class,
                        target_class=target_class,
                        dataset_name=self.dataset_name,
                        reconstruction_mode=True 
                    )
                    simulation_model.zero_grad(set_to_none=True)
            
            print("\n--> Simulazione attacco terminata.")
            return 
        
        print('\n===> Simulation started (TRAINING MODE)...')
        fl_dfndr = FLDefender(self.num_peers)
        
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        
        start_round = 0
        if resume:
            print('Loading last saved checkpoint..')
            checkpoint = torch.load(dynamic_checkpoint_name, weights_only=False)
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']
            print('>>checkpoint loaded!')
            
        print("\n====>Global model training started...\n")
        for epoch in tqdm(range(start_round, self.global_rounds)):
            print(f'\n | Global training round : {epoch+1}/{self.global_rounds} |\n')
            
            global_state_dict = simulation_model.state_dict()
            selected_peers = self.choose_peers()
            local_weights, local_losses = [], []
            local_models = [] if rule == 'fl_defender' else None
            peers_types = []
            i = 1        
            Peer._performed_attacks = 0
            for peer in selected_peers:
                peers_types.append(self.peers[peer].peer_type)
                
                
                # Instantiate empty local model to avoid deepcopy overhead
                local_model = setup_model(model_architecture=self.model_name, num_classes=self.num_classes, tokenizer=None, embedding_dim=self.embedding_dim)
                local_model.load_state_dict(global_state_dict)
                local_model.to(self.device)

                peer_local_model, peer_loss = self.peers[peer].participant_update(epoch, 
                local_model,
                attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                source_class = source_class, target_class = target_class, 
                dataset_name = self.dataset_name, global_rounds = self.global_rounds)

                # No need to deepcopy the entire model again: we only need the trained weights.
                local_weights.append(peer_local_model.state_dict())
                local_losses.append(peer_loss) 
                if rule == 'fl_defender':
                    local_models.append(peer_local_model)
                i+= 1
            
            loss_avg = sum(local_losses) / len(local_losses)
            print('Average of peers\' local losses: {:.6f}'.format(loss_avg))
            
            # --- APPLICAZIONE DELLA REGOLA DI AGGREGAZIONE ---
            if rule == 'fl_defender':
                cur_time = time.time()
                scores = fl_dfndr.score(simulation_model, 
                                            local_models, 
                                            peers_types = peers_types, 
                                            selected_peers = selected_peers,
                                            epoch = epoch+1,
                                            tau = (1.5*epoch/self.global_rounds))
                
                trust = self.update_score_history(scores, selected_peers, epoch)
                t = time.time() - cur_time
                global_weights = average_weights(local_weights, trust)
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)
            else:
                cur_time = time.time()
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                cpu_runtimes.append(time.time() - cur_time)
            
            simulation_model.load_state_dict(global_weights)           
            if epoch >= self.global_rounds-10:
                last10_updates.append(global_weights) 

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
         
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())
            print('{0:10s} - {1}'.format('Class','Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                if i == source_class:
                    source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
            
            backdoor_asr = 0.0
            if attack_type == 'backdoor':
                backdoor_pattern = Peer._BACKDOOR_PATTERN_CPU[self.dataset_name].to(self.device)
                backdoor_asr = self.test_backdoor(
                    simulation_model,
                    self.device,
                    self.test_loader,
                    backdoor_pattern,
                    source_class,
                    target_class
                )
            print('\nBackdoor ASR', backdoor_asr)
            
            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(), 
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'lf_asr': lf_asr if 'lf_asr' in locals() else 0,
                'backdoor_asr': backdoor_asr if 'backdoor_asr' in locals() else 0
            }
            
            # Salvataggio dinamico ad ogni epoca
            torch.save(state, dynamic_checkpoint_name)

            del local_models
            del local_weights
            gc.collect()
            torch.cuda.empty_cache()

            if epoch == self.global_rounds-1:
                print('Last 10 updates results')
                global_weights = average_weights(last10_updates, 
                np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights) 
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                print('{0:10s} - {1}'.format('Class','Accuracy'))
                lf_asr = 0.0
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                    if i == source_class:
                        source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
                        lf_asr = np.round(r[target_class]/np.sum(r)*100, 2)

                backdoor_asr = 0.0
                if attack_type == 'backdoor':
                    backdoor_pattern = Peer._BACKDOOR_PATTERN_CPU[self.dataset_name].to(self.device)
                    backdoor_asr = self.test_backdoor(
                        simulation_model,
                        self.device,
                        self.test_loader,
                        backdoor_pattern,
                        source_class,
                        target_class
                    )

        final_lf_asr = lf_asr if 'lf_asr' in locals() else 0.0
        final_backdoor_asr = backdoor_asr if 'backdoor_asr' in locals() else 0.0
        final_cpu_runtime = np.mean(cpu_runtimes) if cpu_runtimes else 0.0

        state = {
                'state_dict': simulation_model.state_dict(),
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'lf_asr': final_lf_asr,
                'backdoor_asr': final_backdoor_asr,
                'avg_cpu_runtime': final_cpu_runtime
                }
        
        # Salvataggio finale
        savepath = f"{RESULTS_DIR}/{attack_type}_{self.dataset_name}_{self.model_name}_{self.dd_type}_{rule}_{self.attackers_ratio}.t7"
        
        # Salvataggio effettivo
        print(f"Salvando il modello finale in: {savepath}")
        torch.save(state, savepath)    

        print('\n--- SUMMARY ---')
        print('Global accuracies: ', global_accuracies)
        print('Class {} accuracies: '.format(source_class), source_class_accuracies)
        print('Test loss:', test_losses)
        print('Label-flipping attack succes rate:', final_lf_asr)
        print('Backdoor attack succes rate:', final_backdoor_asr)
        print('Average CPU aggregation runtime:', final_cpu_runtime)