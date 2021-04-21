# flake8: noqa
import json
import numpy
from azureml.core.model import Model
import joblib


def init():
    global LGBM_MODEL
    # Load the model from file into a global object
    #model_path = Model.get_model_path(model_name="porto_seguro_safe_driver_model")
    model_path = Model.get_model_path(model_name="insurance_model.pkl")
    
    LGBM_MODEL = joblib.load(model_path)



def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = LGBM_MODEL.predict(data)
   
    predictions = LGBM_MODEL.predict(data)
    predictions = predictions.astype(int)
    
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['no-claim', 'claim']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    


    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))
    
     # Log the input and output data to appinsights:
    info = {
            "input": raw_data,
            "output": result.tolist()
            }
    print(json.dumps(info))
    
    return ({"result": result.tolist()})


if __name__ == "__main__":
    # Test scoring
    init()
    TEST_ROW = '{"data":[[0,1,8,1,0,0,1,0,0,0,0,0,0,0,12,1,0,0,0.5,0.3,0.610327781,7,1,-1,0,-1,1,1,1,2,1,65,1,0.316227766,0.669556409,0.352136337,3.464101615,0.1,0.8,0.6,1,1,6,3,6,2,9,1,1,1,12,0,1,1,0,0,1],[4,2,5,1,0,0,0,0,1,0,0,0,0,0,5,1,0,0,0.9,0.5,0.771362431,4,1,-1,0,0,11,1,1,0,1,103,1,0.316227766,0.60632002,0.358329457,2.828427125,0.4,0.5,0.4,3,3,8,4,10,2,7,2,0,3,10,0,0,1,1,0,1]]}'  # NOQA: E501
    PREDICTION = run(TEST_ROW, {})
    print("Test result: ", PREDICTION)
