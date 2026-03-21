def contains_class(dataset, source_class):
    """
    Verifica se un dataset locale contiene almeno un'immagine della classe 'source'.
    Utilizzato per assegnare il ruolo di 'attacker' solo ai client che possiedono 
    effettivamente le immagini da avvelenare.
    """
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == source_class:
            return True
    return False