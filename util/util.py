import pickle
'''functions for saving and loading models'''
def save_model(dir: str, model) -> None:
    pickle.dump(model, open(dir, 'wb'))

def load_model(dir: str, model) -> None:
    return pickle.load(model, open(dir, 'rb'))
