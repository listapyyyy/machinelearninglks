import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::<account-id>:role/<your-sagemaker-role>'   # GANTI dengan ARN role Anda

model_data = 's3://technomart-s3-test/models/recommender_model.tar.gz'

model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point='inference.py',
    source_dir='.',                    # direktori tempat inference.py & requirements.txt
    dependencies=['requirements.txt'],
    framework_version='1.2-1',
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='recommender-endpoint'
)

print(f"Endpoint {predictor.endpoint_name} deployed.")