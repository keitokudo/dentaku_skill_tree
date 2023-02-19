import itertools

import torch
import pytorch_lightning as pl

from dentaku_t5_tokenizer import DentakuT5Tokenizer

class DentakuT5Base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("pl_module_setting")

        parser.add_argument("--num_beams", type=int, required=True)
        parser.add_argument("--max_length", type=int, required=True)
        
        parser.add_argument("--dropout_rate", type=float)

        parser.add_argument("--num_heads", type=int)
        parser.add_argument("--num_layers", type=int)
        parser.add_argument("--num_decoder_layers", type=int)


        #Adafactor setting
        parser.add_argument("--eps0", type=float, default=1e-30)
        parser.add_argument("--eps1", type=float, default=1e-3)
        parser.add_argument("--clip_threshold", type=float, default=1.0)
        parser.add_argument("--decay_rate", type=float, default=-0.8)
        parser.add_argument("--beta1", type=float)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        
        
        parser.add_argument("--model_name_or_path", help="Select model name or path", type=str, required=True)
        parser.add_argument("--tokenizer_name_or_path", help="Select model name or path", type=str, required=True)
        
        parser.add_argument("--from_scratch", help="Select whether to use pretrained model", action="store_true")

        parser.add_argument("--calc_answer_only_accuracy", help="Specify whether to calculate answer only accuracy", action="store_true")
        return parser
    
    def __init__(self):
        super().__init__()
        self.test_data_idx = 0
        
    def _post_init(self):
        self.tokenizer = DentakuT5Tokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path
        )
        
    def configure_optimizers(self):
        raise NotImplementedError()

        
    def training_step(self, batch, batch_idx=None):
        raise NotImplementedError()


    def overwrite_model_config(self):
        if self.hparams.dropout_rate is not None:
            self.model_config.dropout_rate = self.hparams.dropout_rate

        if self.hparams.num_decoder_layers is not None:
            self.model_config.num_decoder_layers = self.hparams.num_decoder_layers

        if self.hparams.num_layers is not None:
            self.model_config.num_layers = self.hparams.num_layers

        if self.hparams.num_heads is not None:
            self.model_config.num_heads = self.hparams.num_heads

        
    def validation_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)        
        decode_results = self.decode_and_eval(batch, test=False)
        self.log("valid_loss", output.loss.item(), on_step=False, on_epoch=True)

        decode_results.pop("sample_decoded_text")
        return decode_results

    
    def validation_epoch_end(self, outputs):
        collects = sum(d["collects"] for d in outputs)
        counts = sum(d["counts"] for d in outputs)
        self.log("valid_accuracy", collects / counts)
        
    
    def test_step(self, batch, batch_idx=None):
        output = self.model(**batch, output_hidden_states=False)
        decode_results = self.decode_and_eval(batch, test=True)
        
        self.log(
            f"test_loss_{self.test_data_idx}",
            output.loss.item(),
            on_step=False,
            on_epoch=True
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
            

    def decode_and_eval(self, batch, test=False):
        torch.use_deterministic_algorithms(False)
        answer_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_beams=self.hparams.num_beams,
            max_length=self.hparams.max_length,
            do_sample=False,
            min_length=0,
            num_beam_groups=1,
            no_repeat_ngram_size=0,
            encoder_no_repeat_ngram_size=0,
            length_penalty=0.0,
        )
        torch.use_deterministic_algorithms(True)
        answer_ids = answer_ids.cpu()
        

        if not test:
            sample_decoded_texts = None
        else:
            sample_decoded_texts = [
                self.tokenizer.decode(
                    ans,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                for ans in answer_ids
            ]
            
            
        
        collects, counts = self.calc_accuracy(batch, answer_ids)      

        return {
            "collects": collects,
            "counts": counts,
            "sample_decoded_text": sample_decoded_texts
        }



    
    def calc_accuracy(self, batch, answer_ids):
        answer_ids = answer_ids[:, 1:]
        batch_labels = batch['labels'].masked_fill(
            batch["labels"] == -100,
            self.tokenizer.pad_token_id
        )
        label_length = batch_labels.size(-1)
        answer_length = answer_ids.size(-1)
        
        #大きい方に合わせてpadする
        if label_length > answer_length:
            answer_ids = self.tokenizer.pad({"input_ids" : answer_ids.tolist()}, padding='max_length', max_length=label_length, return_tensors="pt")["input_ids"]
        else:
            batch_labels = self.tokenizer.pad({"input_ids" : batch_labels.tolist()}, padding='max_length', max_length=answer_length, return_tensors="pt")["input_ids"]

        assert answer_ids.size() == batch_labels.size(), "Size is different..."
        
        collects = torch.sum(
            torch.all(
                torch.eq(
                    answer_ids.cpu(),
                    batch_labels.cpu()
                ),
                dim = -1
            ),
            dim = 0
        ).item()
        counts = answer_ids.size(0)
        return collects , counts
    
