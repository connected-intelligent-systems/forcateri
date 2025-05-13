from .dartsmodeladapter import DartsModelAdapter
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries as DartsTimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
import logging
from ..modelexceptions import InvalidModelTypeError, ModelAdapterError
from darts.utils.likelihood_models import QuantileRegression
from ...data.adapterinput import AdapterInput
from typing import List,Optional, Tuple
import pandas as pd

class DartsTCNModel(DartsModelAdapter):
    def __init__(self, *args,**kwargs):
        """
        Initializes the Darts TCNModel with specified parameters and scalers.
        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments to configure the TCNModel. Supported keys include:
            - input_chunk_length (int): Length of the input sequence. Default is 7.
            - output_chunk_length (int): Length of the output sequence. Default is 5.
            - kernel_size (int): Size of the convolutional kernel. Default is 3.
            - num_filters (int): Number of filters in the convolutional layers. Default is 32.
            - dilation_base (int): Base of the dilation factor. Default is 2.
            - num_layers (int): Number of convolutional layers. Default is 3.
            - dropout (float): Dropout rate for regularization. Default is 0.1.
            - weight_norm (bool): Whether to apply weight normalization. Default is True.
            - n_epochs (int): Number of training epochs. Default is 100.
            - batch_size (int): Batch size for training. Default is 32.
            - optimizer_kwargs (dict): Additional arguments for the optimizer. Default is {'lr': 1e-3}.
            - random_state (int, optional): Random seed for reproducibility. Default is None.
            - likelihood (Likelihood, optional): Likelihood model for probabilistic forecasting. 
              Default is QuantileRegression([0.1, 0.5, 0.9]).
        Attributes
        ----------
        model : TCNModel
            The initialized TCNModel with the specified parameters.
        scaler_target : Scaler
            Scaler for the target variable.
        scaler_cov : Scaler
            Scaler for the covariates.
        """
        
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
        """
        Fits the model using the provided training and validation data.

        Args:
            train_data (List[AdapterInput]): The training data to be used for fitting the model.
            val_data (Optional[List[AdapterInput]]): The validation data to be used for evaluating the model during training. 
                This parameter is optional and can be None.
            **kwargs: Additional keyword arguments to be passed to the parent class's fit method.

        Raises:
            ModelAdapterError: If the model fitting process fails due to invalid parameters or other issues.

        Logs:
            An error message is logged if the model fitting process fails.
        """


        try:
            super().fit(train_data=train_data,val_data=val_data,**kwargs)

        except ModelAdapterError as e:
            logging.error("Failed to fit a model, check the model params")
            raise ModelAdapterError(f"Failed to fit model: {e}") 
   
    def convert_input(self,data:List[AdapterInput]) -> Tuple[List[DartsTimeSeries], List[DartsTimeSeries], List[DartsTimeSeries], Optional[pd.DataFrame]]:
        """
        Converts the input data into the required format for the model, applying scaling transformations
        to the target and observed time series.

        Args:
            data (List[AdapterInput]): A list of AdapterInput objects containing the input data.

        Returns:
            Tuple[List[DartsTimeSeries], List[DartsTimeSeries], List[DartsTimeSeries], Optional[pd.DataFrame]]:
                - target: A list of scaled DartsTimeSeries objects representing the target time series.
                - known: A list of DartsTimeSeries objects representing the known covariates.
                - observed: A list of scaled DartsTimeSeries objects representing the observed covariates.
                - static: An optional pandas DataFrame containing static covariates, if available.
        """

        target, known, observed , static  = super().convert_input(data)
        target = self.scaler_target.fit_transform(target)
        observed = self.scaler_cov.fit_transform(observed)
        return target, known, observed , static  
    
    def predict(self,**kwargs):
        raise NotImplementedError("Predict method is not implemented yet.") 
    
    def tune(self,train_data:List[AdapterInput],val_data:Optional[List[AdapterInput]],**kwargs):
        raise NotImplementedError("Tune method is not implemented yet.")

    