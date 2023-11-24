import torch
import torch.nn as nn
import numpy as np
from codex.Enhancer.intergate_mechanism import Intergate

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, device: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self._device = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=self._device))
            torch.nn.init.xavier_normal_(nn_param)  # initalize
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)), nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self._device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class GRU(torch.nn.Module):
    def __init__(self, args, num_units, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, memory_dim=16):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self.in_dim = args.input_dim
        self._use_gc_for_ru = use_gc_for_ru
        self._memory_dim = 16
        self._device = device
        self._fc_params = LayerParams(self, 'fc', self._device)

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._fc(inputs, r * hx, self._num_units).reshape(-1, self._num_nodes * self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        # print('inputs.shape:{}'.format(inputs.shape))
        # print('state.shape:{}'.format(state.shape))
        inputs_and_state = torch.cat([inputs, state], dim=1)
        input_size = inputs_and_state.shape[-1]
        # print('in_size:{}, out_size:{}'.format(input_size, output_size))
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        # print('out_size:{}, bias_start:{}'.format(output_size, bias_start))
        biases = self._fc_params.get_biases(output_size, bias_start)
        # print('biases.shape:{}'.format(biases.shape))
        value = value + biases
        return value


class Seq2SeqAttrs:
    def __init__(self, args):
        #self.adj_mx = adj_mx
        self.max_diffusion_step = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.filter_type = args.filter_type
        self.num_nodes = args.num_nodes
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_units = args.rnn_units
        self.output_dim = args.output_dim
        self.hidden_state_size = self.num_nodes * self.rnn_units

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.seq_len = args.lookback  # for the encoder
        self.device = args.device
        self.gru_layers = nn.ModuleList(
            [GRU(args, self.rnn_units, self.max_diffusion_step, self.num_nodes, device=self.device,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])


    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, gru_layer in enumerate(self.gru_layers):
            next_hidden_state = gru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.output_dim = args.output_dim
        self.horizon = args.horizon  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.device = args.device
        self.gru_layers = nn.ModuleList(
            [GRU(args, self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=self.device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, gru_layer in enumerate(self.gru_layers):
            next_hidden_state = gru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class GRU_model(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        super(GRU_model, self).__init__()
        Seq2SeqAttrs.__init__(self, args)
        self.args = args
        self.encoder_model = EncoderModel(args)
        self.decoder_model = DecoderModel(args)
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning
        self.seq_len = args.lookback
        self.batch_size= args.batch_size
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.Intergrate = Intergate(args).to(self.device)
        self.output_dim = args.output_dim
        self.dataset = args.dataset


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)

            decoder_input = decoder_output

            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning and labels is not None:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, transform_data, invariant_pattern=None, variant_pattern=None, intervene=None, labels=None, batches_seen=None, abla=False):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        b, n, t, d = transform_data.shape
        if intervene:
            pred_list = []
            feature_list = self.Intergrate(transform_data, variant_pattern, invariant_pattern, intervene)
            for feature in feature_list:
                feature = feature.permute(2, 0, 1, 3).reshape(t, b, -1)
                encoder_hidden_state = self.encoder(feature)
                pred = self.decoder(encoder_hidden_state).squeeze(dim=0)
                pred_list.append(pred)
            return pred_list
        else:
            feature = self.Intergrate(transform_data, variant_pattern, invariant_pattern, intervene)
            feature = feature.permute(2, 0, 1, 3).reshape(t, b, -1)
            encoder_hidden_state = self.encoder(feature)
            pred = self.decoder(encoder_hidden_state).squeeze(dim=0)
            return pred
