import os
import six
import pickle
import paddle

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save(state_dicts, file_name):

    def convert(state_dict):
        model_dict = {}
        
        for k, v in state_dict.items():
            if isinstance(v, (paddle.framework.Variable, paddle.imperative.core.VarBase)):
                model_dict[k] = v.numpy()
            else:
                model_dict[k] = v

        return model_dict

    final_dict = {}
    for k, v in state_dicts.items():
        if isinstance(v, (paddle.framework.Variable, paddle.imperative.core.VarBase)):
            final_dict = convert(state_dicts)
            break
        elif isinstance(v, dict):
            final_dict[k] = convert(v)
        else:
            final_dict[k] = v
    
    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f, protocol=2)


def load(file_name):
    with open(file_name, 'rb') as f:
        state_dicts = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    return state_dicts

    

    