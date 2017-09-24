import jarvis
from scen1.inception_model_getter import get_inception_model
from scen1.inception_featurizer import inception_featurizer
from scen1.training_data_getter import get_training_data

getter_incmod = jarvis.Action(get_inception_model)
inception_model = jarvis.Artifact('classify_image_graph_def.pb', getter_incmod)

getter_tr = jarvis.Action(get_training_data)
tr_data = jarvis.Artifact('training_data.pkl', getter_tr)

featurize = jarvis.Action(inception_featurizer, [inception_model, tr_data])
feat_data = jarvis.Artifact('featurized_data.pkl', featurize) 

feat_data.pull()