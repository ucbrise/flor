import project
from training_data_getter import get_training_data
from inception_model_getter import get_inception_model
from inception_featurizer import inception_featurizer
from split import split

getter_incmod = project.Action(get_inception_model)
inception_model = project.Artifact('classify_image_graph_def.pb', getter_incmod)

getter_tr = project.Action(get_training_data)
tr_data = project.Artifact('training_data.pkl', getter_tr)

featurize = project.Action(inception_featurizer, [inception_model, tr_data])
feat_data = project.Artifact('featurized_data.pkl', featurize)

do_split = project.Action(split, [feat_data, ])
tr_feat_data = project.Artifact('training_featurized_data.pkl', do_split)
te_feat_data = project.Artifact('testing_featurized_data.pkl', do_split)

tr_feat_data.pull()
tr_feat_data.plot()