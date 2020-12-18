import torch
import soundfile as sf
import torch.nn.functional as F
import itertools as it
from fairseq.data import Dictionary
from fairseq.data.data_utils import post_process
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecEncoder, Wav2VecCtc
from wav2letter.decoder import CriterionType
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Wav2vec-2.0 Recognize')
    parser.add_argument('w2v_path', type=str,
                        help='path of pre-trained wav2vec-2.0 model')
    parser.add_argument('wav_path', type=str,
                        help='path of wave file')
    parser.add_argument('--target_dict_path', type=str,
                        default='dict.ltr.txt',
                        help='path of target dict (dict.ltr.txt)')
    return parser.parse_args()

def base_architecture(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)
    return args

class W2lDecoder(object):
    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.asg_transitions = None

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)

        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))


# from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = list()

        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)

        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}] for b in range(B)
        ]

class Wav2VecPredictor:
    def __init__(self, w2v_path, target_dict_path):
        self._target_dict = Dictionary.load(target_dict_path)
        self._generator = W2lViterbiDecoder(self._target_dict)
        self._model = self._load_model(w2v_path, self._target_dict)
        self._model.eval()

    def _get_feature(self, filepath):
        def postprocess(feats, sample_rate):
            if feats.dim() == 2:
                feats = feats.mean(-1)

            assert feats.dim() == 1, feats.dim()

            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
            return feats

        wav, sample_rate = sf.read(filepath)
        feats = torch.from_numpy(wav).float()
        feats = postprocess(feats, sample_rate)
        return feats

    def _load_model(self, model_path, target_dict):
        w2v = torch.load(model_path)

        # Without create a FairseqTask
        args = base_architecture(w2v["args"])
        model = Wav2VecCtc(args, Wav2VecEncoder(args, target_dict))
        model.load_state_dict(w2v["model"], strict=True)
        return model

    def predict(self, wav_path):
        sample = dict()
        net_input = dict()

        feature = self._get_feature(wav_path)
        net_input["source"] = feature.unsqueeze(0)

        padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)

        net_input["padding_mask"] = padding_mask
        sample["net_input"] = net_input

        with torch.no_grad():
            hypo = self._generator.generate([ self._model ], sample, prefix_tokens=None)

        hyp_pieces = self._target_dict.string(hypo[0][0]["tokens"].int().cpu())
        return post_process(hyp_pieces, 'letter')

if __name__ == '__main__':
    args = parse_args()
    model = Wav2VecPredictor(args.w2v_path, args.target_dict_path)
    print(model.predict(args.wav_path))
