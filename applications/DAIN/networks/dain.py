import paddle.fluid as fluid
import resblock
import pwcnet


class DAIN(fluid.dygraph.Layer):
    def __init__(self, channel=3, filter_size=4, timestep=0.5, training=True):
        # base class initialization
        super(DAIN, self).__init__()

        self.filter_size = filter_size
        self.training = training
        self.timestep = timestep
        assert (timestep == 0.5)
        self.numFrames = int(1.0 / timestep) - 1

        ctx_ch = 3 * 64 + 3
        inplanes = 3 + 3 + 3 + 2 * 1 + 2 * 2 + 16 * 2 + 2 * ctx_ch

        self.rectifyNet = resblock.__dict__['MultipleBasicBlock_4'](inplanes,
                                                                    64)
        self.flownets = pwcnet.__dict__['pwc_dc_net']()
        self.div_flow = 20.0

    def forward(self, input):
        """
        Parameters
        ----------
        input: shape (3, batch, 3, width, height)
        -----------
        """
        losses = []
        offsets = []
        '''
            STEP 1: sequeeze the input
        '''
        if self.training == True:

            assert input.shape[0] == 3
            input_0 = input[0]
            input_1 = input[1]
            input_2 = input[2]
        else:
            # print(input.shape[0])
            assert input.shape[0] == 2
            input_0 = input[0]
            input_2 = input[1]

        #prepare the input data of current scale
        cur_input_0 = input_0
        if self.training == True:
            cur_input_1 = input_1
        cur_input_2 = input_2
        '''
            STEP 3.2: concatenating the inputs.
        '''
        cur_offset_input = fluid.layers.concat([cur_input_0, cur_input_2],
                                               axis=1)
        '''
            STEP 3.3: perform the estimation
        '''
        time_offsets = [
            kk * self.timestep for kk in range(1, 1 + self.numFrames, 1)
        ]

        cur_offset_outputs = [
            self.forward_flownets(self.flownets,
                                  cur_offset_input,
                                  time_offsets=time_offsets),
            self.forward_flownets(self.flownets,
                                  fluid.layers.concat(
                                      [cur_input_2, cur_input_0], axis=1),
                                  time_offsets=time_offsets[::-1])
        ]

        cur_offset_output = [cur_offset_outputs[0][0], cur_offset_outputs[1][0]]

        # Warp image use warp-op in PWC-Net
        ref0 = self.flownets.warp_nomask(cur_input_0, cur_offset_output[0])
        ref2 = self.flownets.warp_nomask(cur_input_2, cur_offset_output[1])
        cur_output = (ref0 + ref2) / 2.0

        rectify_input = fluid.layers.concat([
            cur_output, ref0, ref2, cur_offset_output[0], cur_offset_output[1]
        ],
                                            axis=1)

        cur_output_rectified = self.rectifyNet(rectify_input) + cur_output
        '''
            STEP 3.5: for training phase, we collect the variables to be penalized.
        '''
        if self.training == True:
            losses += [cur_output - cur_input_1]
            losses += [cur_output_rectified - cur_input_1]
            offsets += [cur_offset_output]
        '''
            STEP 4: return the results
        '''
        if self.training == True:
            # if in the training phase, we output the losses to be minimized.
            # return losses, loss_occlusion
            return losses, offsets
        else:
            cur_outputs = [cur_output, cur_output_rectified]
            return cur_outputs, cur_offset_output

    def forward_flownets(self, model, input, time_offsets=None):

        if time_offsets == None:
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass
        # this is a single direction motion results, but not a bidirectional one
        temp = model(input)

        # single direction to bidirection should haven it.
        temps = [
            self.div_flow * temp * time_offset for time_offset in time_offsets
        ]
        # nearest interpolation won't be better i think
        temps = [fluid.layers.resize_bilinear(temp, scale=4) for temp in temps]
        return temps
