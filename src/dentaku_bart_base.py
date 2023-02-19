from dentaku_t5_base import DentakuT5Base
from dentaku_bart_tokenizer import DentakuBartTokenizer

class DentakuBartBase(DentakuT5Base):
    def _post_init(self):
        self.tokenizer = DentakuBartTokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path
        )
        
    def overwrite_model_config(self):
        if self.hparams.dropout_rate is not None:
            self.model_config.dropout = self.hparams.dropout_rate
            
        if self.hparams.num_decoder_layers is not None:
            self.model_config.decoder_layers = self.hparams.num_decoder_layers

        if self.hparams.num_layers is not None:
            self.model_config.encoder_layers = self.hparams.num_layers

        if self.hparams.num_heads is not None:
            self.model_config.encoder_attention_heads = self.hparams.num_heads
            self.model_config.decoder_attention_heads = self.hparams.num_heads
 
