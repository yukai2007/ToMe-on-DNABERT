from torch import nn
from tome.utils import parse_r
import torch
import sys
from torch.nn import DataParallel
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
sys.path.append('/root/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT-2-117M/')
from d064dece8a8b41d9fb8729fbe3435278786931f1.bert_layers import BertForSequenceClassification,ToMeBertUnpadSelfAttention, ToMeBertLayer,ToMeBertEncoder,ToMeBertUnpadAttention
import torch
from typing import Optional

def make_tome_class(transformer_class):
    class ToMeBertForSequenceClassification(transformer_class):
        def __init__(self, config,ToMeList:torch.Tensor=None):
            super().__init__(config)
            self.ToMeList=ToMeList
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True, 
        ):
            # print("ToMeBertForSequenceClassification")
            # print(self.ToMeList)
            self._tome_info["r"] = self.r
            self._tome_info["size"] = [None,None,None,None,None,None,None,None]
            # if input_ids is not None:
            #     # Create a tensor of the same shape as input_ids with all values set to 1
            #     size_tensor = torch.ones_like(input_ids, dtype=torch.float)
                
            #     # Optionally, you can set the padding tokens (if attention_mask is provided) to 0
            #     if attention_mask is not None:
            #         size_tensor = size_tensor * attention_mask.float()  # Masks padding tokens as 0
            #     self._tome_info["size"] = size_tensor
            self._tome_info["source"] = None
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                ToMeList=self.ToMeList,
            )
            last_hidden_state = outputs[0]
            pooled_output = outputs[1]
            # 获取token的大小信息
            size = self._tome_info["size"][torch.cuda.current_device()]
            if size is not None:
                # 确保size和last_hidden_state在同一个设备上
                last_hidden_state = last_hidden_state.to(size.device)  # 将size移到与last_hidden_state相同的设备上
                # print(f"last_hidden_state.shape={last_hidden_state.shape},size.shape={size.shape}")
                # 加权操作：用size对每个位置的token表示进行加权
                # print(f"{torch.cuda.current_device()}:last_hidden_state.shape={last_hidden_state.shape},size.shape={size.shape}")
                weighted_state = last_hidden_state * size  # 广播size到hidden_size维度

                # 然后进行池化操作
                pooled_output = weighted_state.sum(dim=1) / size.sum(dim=1)  # sum and normalize
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            # if torch.cuda.current_device()==0:
            #     print(f"pooled_output.shape={pooled_output.shape}")
            #     print(f"logits.shape={logits.shape}")
            #     print(f"labels.shape={labels.shape}")
            #     print(f"logits={logits}")
            #     print(f"labels={labels}")

            loss = None
            if labels is not None:
                # Compute loss
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = 'regression'
                    elif self.num_labels > 1 and (labels.dtype == torch.long or
                                                labels.dtype == torch.int):
                        self.config.problem_type = 'single_label_classification'
                    else:
                        self.config.problem_type = 'multi_label_classification'

                if self.config.problem_type == 'regression':
                    loss_fct = nn.MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == 'single_label_classification':
                    loss_fct = nn.CrossEntropyLoss()
                    # if torch.cuda.current_device()==0:
                        # print(f"logits.view(-1, self.num_labels).shape={(logits.view(-1, self.num_labels)).shape},labels.view(-1).shape={(labels.view(-1)).shape}")
                    loss = loss_fct(logits.view(-1,self.num_labels),
                                    labels.view(-1))
                elif self.config.problem_type == 'multi_label_classification':
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs[0],
                attentions=None,
            )

    return ToMeBertForSequenceClassification


def apply_patch(
    model: BertForSequenceClassification, trace_source: bool = False, prop_attn: bool = False, r: int = 0,ToMeList:torch.Tensor=None
):
    # print("apply_patch")
    # print(ToMeList)
    ToMeBertForSequenceClassification = make_tome_class(BertForSequenceClassification)
    model.__class__ = ToMeBertForSequenceClassification
    model.r = r
    model.ToMeList=ToMeList
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    # print("Trying to match modules")
    for layer in model.bert.encoder.layer:
        for module in layer.modules():
            if module.__class__.__name__ == "BertLayer":
                module._tome_info = model._tome_info
            elif module.__class__.__name__ == "BertUnpadSelfAttention":
                module._tome_info = model._tome_info
            elif module.__class__.__name__ == "BertUnpadAttention":
                module._tome_info = model._tome_info
            elif module.__class__.__name__ == "ToMeBertLayer":
                module._tome_info = model._tome_info
            elif module.__class__.__name__ == "ToMeBertUnpadSelfAttention":
                module._tome_info = model._tome_info
            elif module.__class__.__name__ == "ToMeBertUnpadAttention":
                module._tome_info = model._tome_info
    if model.bert.encoder.__class__.__name__ == "BertEncoder":
        model.bert.encoder._tome_info = model._tome_info
    return model
