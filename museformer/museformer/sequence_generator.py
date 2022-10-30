from fairseq.sequence_generator import SequenceGenerator


class MuseformerSequenceGenerator(SequenceGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = self.model.single_model.decoder.valid_dictionary_len
        self.beam_size = min(self.beam_size, self.vocab_size - 1)
