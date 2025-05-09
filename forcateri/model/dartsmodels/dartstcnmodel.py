from .dartsmodeladapter import DartsModelAdapter
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries as DartsTimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
import logging
from ..modelexceptions import InvalidModelTypeError, ModelAdapterError
from darts.utils.likelihood_models import QuantileRegression
from ...data.adapterinput import AdapterInput
from typing import List,Optional

class DartsTCNModel(DartsModelAdapter):
    def __init__(self, *args,**kwargs):
        
        super().__init__(*args,**kwargs)
        self.model = TCNModel(
            input_chunk_length=kwargs.get('input_chunk_length', 7),
            output_chunk_length=kwargs.get('output_chunk_length', 5),
            kernel_size=kwargs.get('kernel_size', 3),
            num_filters=kwargs.get('num_filters', 32),
            dilation_base=kwargs.get('dilation_base', 2),
            num_layers=kwargs.get('num_layers', 3),
            dropout=kwargs.get('dropout', 0.1),
            weight_norm=kwargs.get('weight_norm', True),
            n_epochs=kwargs.get('n_epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            optimizer_kwargs=kwargs.get('optimizer_kwargs', {'lr': 1e-3}),
            random_state=kwargs.get('random_state', None),
            likelihood=kwargs.get('likelihood', QuantileRegression([0.1, 0.5, 0.9])),
        )

        self.scaler_target = Scaler()
        self.scaler_cov = Scaler()
    
    def fit(self,train_data:List[AdapterInput],val_data:Optional[List[AdapterInput]],**kwargs):
        super().fit(train_data=train_data,val_data=val_data,**kwargs)
        
        scaled_target = self.scaler_target.fit_transform(self.fit_args['target'])
        scaled_cov = self.scaler_cov.fit_transform(self.fit_args['past_covariates'])
        scaled_val_series = self.scaler_target.transform(self.fit_args['val_series'])
        scaled_val_cov = self.scaler_cov.transform(self.fit_args['val_past_covariates'])
        

        try:
            self.model.fit(
                series=scaled_target,
                past_covariates=scaled_cov,
                val_series=scaled_val_series, 
                val_past_covariates=scaled_val_cov,
                verbose=True
            )
        except ModelAdapterError as e:
            logging.error("Failed to fit a model, check the model params")
            raise ModelAdapterError(f"Failed to fit model: {e}") 
   
    def predict(self,**kwargs):
        raise NotImplementedError("Predict method is not implemented yet.") 

    