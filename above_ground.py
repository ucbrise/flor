def __tags_equal__(groundtag, mytag):
    groundtagprime = {}
    for kee in groundtag:
        groundtagprime[kee] = {}
        for kii in groundtag[kee]:
            if kii != 'id':
                groundtagprime[kee][kii] = groundtag[kee][kii]
    return groundtagprime == mytag

def newExperimentVersion(xp_state):
    # -- caution with fixed values like 'jarvisExperiment', allowing for early Ground Ref prototype

    # The name of this experiment is in a tag in the nodeVersion of 'jarvisExperiment'
    latest_experiment_node_versions = [x for x in xp_state.gc.getNodeLatestVersions('jarvisExperiment')
                                       if xp_state.gc.getNodeVersion(x).get_tags()['experimentName'][
                                           'value'] == xp_state.EXPERIMENT_NAME]

    # This experiment may have previous versions, then the most recents are the parents
    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisExperiment').get_id(),
                                       tags={
                                           'experimentName': {
                                               'key': 'experimentName',
                                               'value': xp_state.EXPERIMENT_NAME,
                                               'type': 'STRING'
                                           }},
                                       parentIds=latest_experiment_node_versions)

def newTrialVersion(xp_state):
    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisTrial').get_id())

def newLiteralVersion(xp_state, literalName, literalValue):

    my_tag = {     'literalName' : {
                         'key' : 'literalName',
                         'value' : literalName,
                         'type' : 'STRING'
                     },
                     'literalValue' : {
                         'key': 'literalValue',
                         'value' : str(literalValue),
                         'type' : 'STRING'
                     }
                 }

    candidate_nvs = [xp_state.gc.getNodeVersion(str(x)) for x in xp_state.gc.getNodeLatestVersions('jarvisLiteral')
                     if __tags_equal__(xp_state.gc.getNodeVersion(str(x)).get_tags(), my_tag)]
    assert len(candidate_nvs) <= 1

    if len(candidate_nvs) == 1:
        return candidate_nvs[0]
    else:
        return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisLiteral').get_id(),
                                         tags = my_tag)

def newArtifactVersion(xp_state, artifactName, md5hash):
    # Connect artifact versions to parents offline
    # What's important at this level is the tags
    # What artifact meta-data do we care about

    my_tag = {
                   'artifactName': {
                       'key': 'artifactName',
                       'value': artifactName,
                       'type': 'STRING'
                   },
                    'artifactMD5': {
                        'key' : 'artifactMD5',
                        'value' : md5hash,
                        'type' : 'STRING'
                    }
               }

    candidate_nvs = [xp_state.gc.getNodeVersion(str(x)) for x in xp_state.gc.getNodeLatestVersions('jarvisArtifact')
                     if __tags_equal__(xp_state.gc.getNodeVersion(str(x)).get_tags(), my_tag)]
    assert len(candidate_nvs) <= 1

    if len(candidate_nvs) == 1:
        return candidate_nvs[0]
    else:
        return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisArtifact').get_id(),
                                       tags=my_tag)

def newExperimentTrialEdgeVersion(xp_state, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisExperimentjarvisTrial')

def newTrialLiteralEdgeVersion(xp_state, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisTrialjarvisLiteral')

def newTrialArtifactEdgeVersion(xp_state, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisTrialjarvisArtifact')

def __newEdgeVersion__(xp_state, fromNv, toNv, edgeKey):
    return xp_state.gc.createEdgeVersion(xp_state.gc.getEdge(edgeKey).get_id(),
                                         fromNv.get_id(),
                                         toNv.get_id())

