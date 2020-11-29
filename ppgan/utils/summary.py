import paddle
import paddle.nn as nn
from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].shape)
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.shape)[1:]
                                                  for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.shape)
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += paddle.prod(
                    paddle.to_tensor(list(module.weight.shape)))
                summary[m_key]["trainable"] = module.weight.stop_gradient
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += paddle.prod(paddle.to_tensor(list(module.bias.shape)))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.LayerList)
                and not (module == model)):
            hooks.append(module.register_forward_post_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [paddle.rand((2, *in_size)) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape",
                                              "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        nb_params = summary[layer]["nb_params"]
        nb_params = nb_params if isinstance(nb_params,
                                            int) else nb_params.numpy().item()
        output_shape = summary[layer]["output_shape"]
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(output_shape),
            "{0:,}".format(nb_params),
        )
        total_params += nb_params
        total_output += np.prod(output_shape)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += nb_params
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024**2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024**2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024**2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
