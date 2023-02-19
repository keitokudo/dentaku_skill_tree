import itertools

import torch
import pytorch_lightning as pl

from dentaku_t5 import DentakuT5

class DentakuT5PL(DentakuT5):
    
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = True

    
    def configure_optimizers(self):
        optimizers, lr_shedulers = super().configure_optimizers()
        assert len(optimizers) == len(lr_shedulers) == 1
        optimizer = optimizers[0]
        lr_sheduler = lr_shedulers[0]
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_sheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)
        self.log("train_loss", output.loss.item(), on_step=True, on_epoch=True)
        return {"loss": output.loss}


    def validation_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)        
        decode_results = self.decode_and_eval(batch, test=False)
        self.log("valid_loss", output.loss.item(), on_step=False, on_epoch=True, sync_dist=True)
        decode_results.pop("sample_decoded_text")
        return decode_results

    
    def validation_epoch_end(self, outputs):
        collects = sum(d["collects"] for d in outputs)
        counts = sum(d["counts"] for d in outputs)
        self.log("valid_accuracy", collects / counts, rank_zero_only=True)
        
    
    def test_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)
        decode_results = self.decode_and_eval(batch, test=True)

        self.log(
            f"test_loss_{self.test_data_idx}",
            output.loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
         
        context_and_question = [
            self.tokenizer.decode(
                input_id,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            for input_id in batch["input_ids"].cpu()
        ]

        decoded_labels = [
            self.tokenizer.decode(
                label,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            for label in batch["labels"].masked_fill(
                    batch["labels"] == -100,
                    self.tokenizer.pad_token_id
            ).cpu()
        ]
        
        assert len(context_and_question) == len(decoded_labels) == len(decode_results["sample_decoded_text"]), "Number of \"Question is not equal number of answer\""
        
        test_results = {}
        test_results.update(decode_results)
        test_results["context_and_question"] = context_and_question
        test_results["decoded_labels"] = decoded_labels    
        return test_results
    
    
    def test_epoch_end(self, outputs):
        collects = sum(d["collects"] for d in outputs)
        counts = sum(d["counts"] for d in outputs)

        if self.trainer.is_global_zero:
            self.log(f"test_accuracy_{self.test_data_idx}", collects / counts)
            self.logger.log_text(
                key=f"test_predictions_{self.test_data_idx}",
                columns=["Context and Question", "Answer", "Prediction"],
                data=list(
                    zip(
                        itertools.chain.from_iterable(
                            d["context_and_question"] for d in outputs
                        ),
                        itertools.chain.from_iterable(
                            d["decoded_labels"] for d in outputs
                        ),
                        itertools.chain.from_iterable(
                            d["sample_decoded_text"] for d in outputs
                        ),
                    )
                ),
            )
            
            if self.hparams.calc_answer_only_accuracy:
                extract_answer_part = lambda answer_with_reasoning_step: \
                    "".join(
                        reversed(
                            list(
                                itertools.takewhile(
                                    lambda s: s != "=",
                                    reversed(answer_with_reasoning_step)
                                )
                            )
                        )
                    ).replace(" ", "")


                corrects = sum(
                    extract_answer_part(label) == extract_answer_part(predict)
                    for label, predict in zip(
                            itertools.chain.from_iterable(
                                d["decoded_labels"] for d in outputs
                            ),
                            itertools.chain.from_iterable(
                                d["sample_decoded_text"] for d in outputs
                            ),
                    )
                )
                counts = sum(len(d["sample_decoded_text"]) for d in outputs)
                self.log(f"test_answer_part_accuracy_{self.test_data_idx}", corrects / counts)

