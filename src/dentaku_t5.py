
import torch
from transformers import T5ForConditionalGeneration, T5Config
from logzero import logger

from transformers import Adafactor

from dentaku_t5_tokenizer import DentakuT5Tokenizer
from t5_lr_scheduler import T5InverseSquareRootScheduler, NoSheduler
from dentaku_t5_base import DentakuT5Base

class DentakuT5(DentakuT5Base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = DentakuT5Base.add_model_specific_args(parent_parser)
        parser.add_argument("--lr", type=float, required=True)
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            choices=["inverse_square_root_scheduler", "constant_scheduler"],
            required=True,
        )
        parser.add_argument("--num_warmup_steps", type=int)
        return parent_parser
    
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(config)
        
        if self.hparams.from_scratch:
            if not (self.hparams.model_name_or_path.startswith("google/t5-v1_1") or self.hparams.model_name_or_path.startswith("t5-")):
                logger.warning(
                    "Although local pretrained model was specified, \"from_scratch\" is also selleted"
                )
                
            self.model_config = T5Config.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = T5ForConditionalGeneration(config=self.model_config)
            logger.info("Learning from scrach!")
        else:

            self.model_config = T5Config.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path,
                config=self.model_config
            )
            logger.info(f"Load pretrained model from \"{self.hparams.model_name_or_path}\"")
        
        self._post_init()

        
    def configure_optimizers(self):
        optimizer = Adafactor(
            self.model.parameters(),
            lr=self.hparams.lr,
            eps=(self.hparams.eps0, self.hparams.eps1),
            clip_threshold=self.hparams.clip_threshold,
            decay_rate=self.hparams.decay_rate,
            beta1=self.hparams.beta1,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )


        if self.hparams.lr_scheduler == "inverse_square_root_scheduler":
            assert self.hparams.num_warmup_steps is not None
            lr_sheduler = T5InverseSquareRootScheduler(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
            )
        elif self.hparams.lr_scheduler == "constant_scheduler":
            lr_sheduler = NoSheduler(
                optimizer=optimizer,
            )
        else:
            raise ValueError()
        
        return [optimizer], [lr_sheduler]
    

        
    def training_step(self, batch, batch_idx=None):
        optimizer = self.optimizers(use_pl_optimizer=False)
        lr_sheduler = self.lr_schedulers()
        
        output = self.model(**batch, output_hidden_states=False)
        
        optimizer.zero_grad()
        self.manual_backward(output.loss)
        optimizer.step()
        lr_sheduler.step()
        self.log("train_loss", output.loss.item(), on_step=True, on_epoch=True)
        


