from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from typing import Dict, List, Optional, Union
from typing_extensions import TypedDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
import numpy as np
import torch
from src.utils import load_pickle


def prepare_target_label(
        model: torch.nn.Module, target_label: Union[str, int]
) -> Union[str, int]:
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(
            f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.'
        )
    return target_label


def classify_texts(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        texts: List[str],
        second_texts: Optional[List[str]] = None,
        target_label: Optional[Union[str, int]] = None
):
    target_label = prepare_target_label(model, target_label)
    inputs = [texts]
    if second_texts is not None:
        inputs.append(second_texts)
    inputs = tokenizer(
        *inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
        if logits.shape[-1] > 1:
            preds = torch.softmax(logits, -1)[:, target_label]
        else:
            preds = torch.sigmoid(logits)[:, 0]
        preds = preds.view(-1).cpu().numpy()
    return preds


@register_task
class ruDetox(Task):

    def __init__(self, conf):
        super().__init__(conf)
        no_load_models = False
        if hasattr(self.conf, "no_load_models"):
            no_load_models = self.conf.no_load_models
        if not no_load_models:
            self.__load_models()

    def __load_models(self):
        self.style_model = AutoModelForSequenceClassification.from_pretrained(
            self.task_conf.style_model_path
        )
        self.style_model.to(self.task_conf.device)
        self.style_tokenizer = AutoTokenizer.from_pretrained(
            self.task_conf.style_model_path
        )

        style_calibration = load_pickle(self.task_conf.calibrations_ru_path)
        self.style_calibration = lambda x: style_calibration.predict(x[:, np.newaxis])

        self.meaning_model = AutoModelForSequenceClassification.from_pretrained(
            self.task_conf.meaning_model_path
        )
        self.meaning_model.to(self.task_conf.device)
        self.meaning_tokenizer = AutoTokenizer.from_pretrained(
            self.task_conf.meaning_model_path
        )

        meaning_calibration = load_pickle(self.task_conf.calibrations_ru_path)
        self.meaning_calibration = lambda x: meaning_calibration.predict(
            x[:, np.newaxis]
        )

        self.cola_model = AutoModelForSequenceClassification.from_pretrained(
            self.task_conf.cola_model_path
        )
        self.cola_model.to(self.task_conf.device)
        self.cola_tokenizer = AutoTokenizer.from_pretrained(
            self.task_conf.cola_model_path
        )

        fluency_calibration = load_pickle(self.task_conf.calibrations_ru_path)
        self.fluency_calibration = lambda x: fluency_calibration.predict(
            x[:, np.newaxis]
        )

    def evaluate_style(self, texts):
        target_label = prepare_target_label(self.style_model, 0)
        scores = classify_texts(
            self.style_model,
            self.style_tokenizer,
            [texts],
            target_label=target_label,
        )
        return float(self.style_calibration(scores))

    def evaluate_meaning(self, original_texts, rewritten_texts):
        target_label = prepare_target_label(self.meaning_model, "paraphrase")
        scores = classify_texts(
            self.meaning_model,
            self.meaning_tokenizer,
            [original_texts],
            [rewritten_texts],
            target_label=target_label,
        )
        return float(self.meaning_calibration(scores))

    def evaluate_cola(self, texts):
        target_label = prepare_target_label(self.cola_model, 1)
        scores = classify_texts(
            self.cola_model,
            self.cola_tokenizer,
            [texts],
            target_label=target_label,
        )
        return float(self.fluency_calibration(scores))

    def aggregation(self) -> Dict:
        return {"sta": mean, "sim": mean, "fl": mean, "j": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = doc_true["inputs"]
        y_pred = self.doc_to_y_pred(doc_pred)
        sta = self.evaluate_style(y_pred)
        sim = self.evaluate_meaning(y_true, y_pred)
        fl = self.evaluate_cola(y_pred)
        j = sta * sim * fl
        return {"sta": sta, "sim": sim, "fl": fl, "j": j}

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            doc = {
                "outputs": self.gold[doc_id]["inputs"],
                "meta": {"id": doc_id}
            }
            res.append(doc)
        return {"data": {self.split: res}}


class InterpolationParams(TypedDict):
    axis: int
    bounds_error: bool
    copy: bool
    fill_value: np.ndarray
    x: np.ndarray
    y: np.ndarray


class CalibratorParams(TypedDict):
    X_min_: float
    X_max_: float
    X_thresholds_: np.ndarray
    y_thresholds_: np.ndarray
    y_max: float
    y_min: float
    f_: interp1d
    increasing_: bool


class CalibratorSignature(TypedDict):
    out_of_bounds: str
    increasing: bool
    y_max: float
    y_min: float


def get_calibrator():
    func_params: InterpolationParams = {
        "axis": 0,
        "bounds_error": False,
        "copy": True,
        "fill_value": np.array(np.nan),
        "x": np.array(
            [
                0.00027650000000001285,
                0.00034440000000002247,
                0.00034559999999994595,
                0.001278699999999966,
                0.0012788000000000244,
                0.0019349499999999908,
                0.001937570000000055,
                0.002486699999999953,
                0.0024885000000000046,
                0.002688770000000007,
                0.0026899599999999912,
                0.02320409999999995,
                0.023239140000000047,
                0.029833699999999963,
                0.02988875000000002,
                0.06509770000000004,
                0.06567305000000001,
                0.1304537,
                0.13059336,
                0.32918555000000005,
                0.32961607000000004,
                0.3611194999999999,
                0.36121990000000004,
                0.44504560000000004,
                0.4457099,
                0.6144483700000001,
                0.6155048599999999,
                0.65356693,
                0.6536861,
                0.65981287,
                0.6614356299999999,
                0.68675756,
                0.68679786,
                0.82462016,
                0.82473633,
                0.8333929,
                0.8343167300000001,
                0.8455976000000001,
                0.84581335,
                0.88914423,
                0.88916642,
                0.92163785,
                0.92167094,
                0.941321425,
                0.941556983,
                0.9431188070000001,
                0.94315397,
                0.9672734000000001,
                0.9672936599999999,
                0.973687772,
                0.973703632,
                0.973939763,
                0.973964272,
                0.97632647,
                0.97637341,
                0.98159888,
                0.981640944,
                0.983554833,
                0.983579926,
                0.990303015,
                0.99031182,
                0.9926355486,
                0.9926386364,
                0.9935122714,
                0.9935136237,
                0.993559784,
                0.993560375,
                0.995515268,
                0.995516817,
                0.9958820036,
                0.9958824734,
                0.9973993234,
                0.9974000666,
                0.9982809896,
                0.9982815925,
                0.9988658517,
                0.9988669739,
                0.9988697876,
                0.9988699318,
                0.998884157,
                0.9988841575,
                0.999144743,
                0.99914501555,
                0.9991559096,
                0.9991560005,
                0.99934383045,
                0.999345196,
                0.99939462944,
                0.9993951180000001,
                0.9994204223,
                0.99942057085,
                0.99951546398,
                0.99951708468,
                0.999558611,
                0.99955952057,
                0.99971446983,
            ],
            dtype=np.float64,
        ),
        "y": np.array(
            [
                0.0,
                0.0,
                0.03896103896103896,
                0.03896103896103896,
                0.04522613065326633,
                0.04522613065326633,
                0.04678362573099415,
                0.04678362573099415,
                0.04918032786885246,
                0.04918032786885246,
                0.06148055207026349,
                0.06148055207026349,
                0.08139534883720931,
                0.08139534883720931,
                0.08627450980392157,
                0.08627450980392157,
                0.1827956989247312,
                0.1827956989247312,
                0.21987951807228914,
                0.21987951807228914,
                0.26666666666666666,
                0.26666666666666666,
                0.296,
                0.296,
                0.3129496402877698,
                0.3129496402877698,
                0.34146341463414637,
                0.34146341463414637,
                0.34615384615384615,
                0.34615384615384615,
                0.4166666666666667,
                0.4166666666666667,
                0.4439461883408072,
                0.4439461883408072,
                0.45,
                0.45,
                0.49056603773584906,
                0.49056603773584906,
                0.5181518151815182,
                0.5181518151815182,
                0.5494505494505495,
                0.5494505494505495,
                0.5570175438596491,
                0.5570175438596491,
                0.6,
                0.6,
                0.6490630323679727,
                0.6490630323679727,
                0.6725352112676056,
                0.6725352112676056,
                0.7142857142857143,
                0.7142857142857143,
                0.7171717171717171,
                0.7171717171717171,
                0.7177700348432056,
                0.7177700348432056,
                0.7183098591549296,
                0.7183098591549296,
                0.7589285714285714,
                0.7589285714285714,
                0.7815315315315315,
                0.7815315315315315,
                0.7972350230414746,
                0.7972350230414746,
                0.8181818181818182,
                0.8181818181818182,
                0.8354037267080745,
                0.8354037267080745,
                0.8714285714285714,
                0.8714285714285714,
                0.8748317631224765,
                0.8748317631224765,
                0.8763157894736842,
                0.8763157894736842,
                0.9046728971962616,
                0.9046728971962616,
                0.9090909090909091,
                0.9090909090909091,
                0.9230769230769231,
                0.9230769230769231,
                0.9404466501240695,
                0.9404466501240695,
                0.9523809523809523,
                0.9523809523809523,
                0.9620060790273556,
                0.9620060790273556,
                0.9647887323943662,
                0.9647887323943662,
                0.975609756097561,
                0.975609756097561,
                0.9760956175298805,
                0.9760956175298805,
                0.979381443298969,
                0.979381443298969,
                1.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    }
    params: CalibratorParams = {
        "X_min_": 0.00027650000000001285,
        "X_max_": 0.99971446983,
        "X_thresholds_": func_params["x"],
        "y_thresholds_": func_params["y"],
        "y_max": 1,
        "y_min": 0,
        "f_": interp1d(**func_params),
        "increasing_": True,
    }
    signature: CalibratorSignature = {
        "out_of_bounds": "clip",
        "increasing": True,
        "y_max": 1,
        "y_min": 0,
    }
    model = IsotonicRegression()
    model.set_params(**signature)
    for param_name, param_value in params.items():
        setattr(model, param_name, param_value)
    return model
