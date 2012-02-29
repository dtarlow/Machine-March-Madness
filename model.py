import numpy as np
import theano
import theano.tensor as T

LOCATION_HOME = 0
LOCATION_AWAY = 1
LOCATION_TOURNEY = 2



def make_simplest_learning_functions(num_teams, D0, H, D, Hp, reg_param1,
                                     reg_param2, xform_params=None):
                            
    # each team just has a mean offensive score and a mean defensive
    # score.  to get a team's predicted score, just average opponents
    # offense with our defense, and vice versa
    
    rng = np.random.RandomState()

    # Initialize latent vectors.  Here they are just single numbers.
    offense_vals = np.asarray(rng.uniform(
        low  = 60,
        high = 70,
        size = (num_teams, 1)), dtype=theano.config.floatX)
    defense_vals = np.asarray(rng.uniform(
        low  = 60,
        high = 70,
        size = (num_teams, 1)), dtype=theano.config.floatX)

    offenses = theano.shared(value=offense_vals, name="offenses")
    defenses = theano.shared(value=defense_vals, name="defenses")

    # inputs
    SUPPORT_BATCH_LEARNING = False  # theano indexing is not cooperating
    if SUPPORT_BATCH_LEARNING:
        team1_ids    = T.ivector("team1_ids")
        team1_locs   = T.ivector("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.ivector("team2_ids")
        team2_locs   = T.ivector("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dvector("team1_scores")
        team2_scores = T.dvector("team2_scores")
    else:   # only support stochastic gradient training
        team1_ids    = T.iscalar("team1_ids")
        team1_locs   = T.iscalar("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.iscalar("team2_ids")
        team2_locs   = T.iscalar("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dscalar("team1_scores")
        team2_scores = T.dscalar("team2_scores")

    # learning parameters
    learning_rate = T.scalar("learning_rate")

    # select appropriate latent vectors
    team1_offenses = offenses[team1_ids,:]
    team1_defenses = defenses[team1_ids,:]
    team2_offenses = offenses[team2_ids,:]
    team2_defenses = defenses[team2_ids,:]

    if SUPPORT_BATCH_LEARNING:
        team1_pred_score = .5 * T.sum(team1_offenses + team2_defenses, axis=1)
        team2_pred_score = .5 * T.sum(team2_offenses + team1_defenses, axis=1)
    else:
        team1_pred_score = .5 * T.sum(team1_offenses + team2_defenses)
        team2_pred_score = .5 * T.sum(team2_offenses + team1_defenses)

    # learning objective
    obj = T.mean(T.sqr(team1_pred_score - team1_scores)) + \
          T.mean(T.sqr(team2_pred_score - team2_scores))

    # Define updates
    params = [offenses, defenses]

    grads  = []
    for p in params:
        g = T.grad(obj, p)
        grads.append(g)

    updates = {}
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    # Create and return theano functions
    out_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs],
                             outputs=[team1_pred_score, team2_pred_score])

    train_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs,
                                team1_scores, team2_scores, learning_rate],
                               outputs=obj,
                               updates=updates)

    return out_fn, train_fn, params


def make_vanilla_pmf_functions(num_teams, D0, H, D, Hp, reg_param1,
                               reg_param2, xform_params=None):
                            
    # D0 : dimension of base latent vectors
    # D  : dimension of transformed latent vectors
    
    rng = np.random.RandomState()

    # Initialize latent vectors
    offense0_vals = np.asarray(rng.uniform(
        low  = -np.sqrt(6./(num_teams+D0)),
        high =  np.sqrt(6./(num_teams+D0)),
        size = (num_teams, D)), dtype=theano.config.floatX)
    defense0_vals = np.asarray(rng.uniform(
        low  = -np.sqrt(6./(num_teams+D0)),
        high =  np.sqrt(6./(num_teams+D0)),
        size = (num_teams, D)), dtype=theano.config.floatX)
        
    offenses0 = theano.shared(value=offense0_vals, name="offenses0")
    defenses0 = theano.shared(value=defense0_vals, name="defenses0")

    # inputs
    SUPPORT_BATCH_LEARNING = False  # theano indexing is not cooperating
    if SUPPORT_BATCH_LEARNING:
        team1_ids    = T.ivector("team1_ids")
        team1_locs   = T.ivector("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.ivector("team2_ids")
        team2_locs   = T.ivector("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dvector("team1_scores")
        team2_scores = T.dvector("team2_scores")
    else:   # only support stochastic gradient training
        team1_ids    = T.iscalar("team1_ids")
        team1_locs   = T.iscalar("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.iscalar("team2_ids")
        team2_locs   = T.iscalar("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dscalar("team1_scores")
        team2_scores = T.dscalar("team2_scores")

    # learning parameters
    learning_rate = T.scalar("learning_rate")

    # select appropriate latent vectors
    team1_offenses = offenses0[team1_ids,:]
    team1_defenses = defenses0[team1_ids,:]
    team2_offenses = offenses0[team2_ids,:]
    team2_defenses = defenses0[team2_ids,:]

    if SUPPORT_BATCH_LEARNING:
        team1_pred_score = T.sum(team1_offenses * team2_defenses, axis=1)
        team2_pred_score = T.sum(team2_offenses * team1_defenses, axis=1)
    else:
        team1_pred_score = T.sum(team1_offenses * team2_defenses)
        team2_pred_score = T.sum(team2_offenses * team1_defenses)
    
    # regularization terms
    reg1 = T.mean(T.sqr(offenses0)) + T.mean(T.sqr(defenses0))

    # learning objective
    obj = T.mean(T.sqr(team1_pred_score - team1_scores)) + \
          T.mean(T.sqr(team2_pred_score - team2_scores)) + \
          reg_param1 * reg1

    # Define updates
    params = [offenses0, defenses0]

    grads  = []
    for p in params:
        g = T.grad(obj, p)
        grads.append(g)

    updates = {}
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    # Create and return theano functions
    out_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs],
                             outputs=[team1_pred_score, team2_pred_score])

    train_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs,
                                team1_scores, team2_scores, learning_rate],
                               outputs=obj,
                               updates=updates)

    return out_fn, train_fn, params


def make_pmf_plus_pace_functions(num_teams, D0, H, D, Hp, reg_param1,
                                 reg_param2, xform_params=None):
                            
    # D0 : dimension of base latent vectors
    # D  : dimension of transformed latent vectors
    
    rng = np.random.RandomState()

    # Initialize latent vectors
    offense0_vals = np.asarray(rng.uniform(
        low  = -np.sqrt(6./(num_teams+D0)),
        high =  np.sqrt(6./(num_teams+D0)),
        size = (num_teams, D)), dtype=theano.config.floatX)
    defense0_vals = np.asarray(rng.uniform(
        low  = -np.sqrt(6./(num_teams+D0)),
        high =  np.sqrt(6./(num_teams+D0)),
        size = (num_teams, D)), dtype=theano.config.floatX)

    # Initialize transformation weights.
    # Different transform for {offense, defense} x {home, away, tourney}
    if xform_params is None:
        paceW1_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(1./(4*D+Hp)),
            high =  np.sqrt(1./(4*D+Hp)),
            size = (4*D,Hp)), dtype=theano.config.floatX)
        paceb1_vals = np.zeros((Hp,), dtype=theano.config.floatX)
        paceW2_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(1./(Hp+1)),
            high =  np.sqrt(1./(Hp+1)),
            size = (Hp,1)), dtype=theano.config.floatX)
        paceb2_vals = np.zeros((1,), dtype=theano.config.floatX)
        mean_score_val = 70*np.ones(1, dtype=theano.config.floatX)

        # Create theano variables
        paceW1 = theano.shared(value=paceW1_vals, name="paceW1")
        paceb1 = theano.shared(value=paceb1_vals, name="paceb1")
        paceW2 = theano.shared(value=paceW2_vals, name="paceW2")
        paceb2 = theano.shared(value=paceb2_vals, name="paceb2")
        mean_score = theano.shared(value=mean_score_val, name="mean_score")
    else:
        # This allows sharing of transform parameters across seasons
        paceW1, paceb1, paceW2, paceb2, mean_score = xform_params
        
    offenses0 = theano.shared(value=offense0_vals, name="offenses0")
    defenses0 = theano.shared(value=defense0_vals, name="defenses0")

    # inputs
    SUPPORT_BATCH_LEARNING = False  # theano indexing is not cooperating
    if SUPPORT_BATCH_LEARNING:
        team1_ids    = T.ivector("team1_ids")
        team1_locs   = T.ivector("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.ivector("team2_ids")
        team2_locs   = T.ivector("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dvector("team1_scores")
        team2_scores = T.dvector("team2_scores")
    else:   # only support stochastic gradient training
        team1_ids    = T.iscalar("team1_ids")
        team1_locs   = T.iscalar("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.iscalar("team2_ids")
        team2_locs   = T.iscalar("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dscalar("team1_scores")
        team2_scores = T.dscalar("team2_scores")

    # learning parameters
    learning_rate = T.scalar("learning_rate")

    # select appropriate latent vectors
    team1_offenses = offenses0[team1_ids,:]
    team1_defenses = defenses0[team1_ids,:]
    team2_offenses = offenses0[team2_ids,:]
    team2_defenses = defenses0[team2_ids,:]

    if SUPPORT_BATCH_LEARNING:
        team1_pred_score0 = T.sum(team1_offenses * team2_defenses, axis=1)
        team2_pred_score0 = T.sum(team2_offenses * team1_defenses, axis=1)
    else:
        team1_pred_score0 = T.sum(team1_offenses * team2_defenses)
        team2_pred_score0 = T.sum(team2_offenses * team1_defenses)

    all_o_and_d_12 = T.concatenate([team1_offenses, team1_defenses,
                                    team2_offenses, team2_defenses])
    all_o_and_d_21 = T.concatenate([team2_offenses, team2_defenses,
                                    team1_offenses, team1_defenses])

    paceH = T.nnet.sigmoid(T.dot(all_o_and_d_12, paceW1) \
                           + T.dot(all_o_and_d_21, paceW1) + paceb1)
    pace = .5 + T.nnet.sigmoid(T.dot(paceH, paceW2) + paceb2)

    team1_pred_score = (team1_pred_score0 + mean_score) * pace
    team2_pred_score = (team2_pred_score0 + mean_score) * pace
    
    # regularization terms
    reg1 = T.mean(T.sqr(offenses0)) + T.mean(T.sqr(defenses0))
    reg2 = T.mean(T.sqr(paceW1)) + T.mean(T.sqr(paceW2))

    # learning objective
    obj = T.mean(T.sqr(team1_pred_score - team1_scores)) + \
          T.mean(T.sqr(team2_pred_score - team2_scores)) + \
          reg_param1 * reg1 + reg_param2 * reg2

    # Define updates
    params = [offenses0, defenses0,
              paceW1, paceb1, paceW2, paceb2, mean_score]

    grads  = []
    for p in params:
        g = T.grad(obj, p)
        grads.append(g)

    updates = {}
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    # Create and return theano functions
    out_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs],
                             outputs=[team1_pred_score, team2_pred_score])

    train_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs,
                                team1_scores, team2_scores, learning_rate],
                               outputs=obj,
                               updates=updates)

    return out_fn, train_fn, params


def make_learning_functions(num_teams, D0, H, D, Hp, reg_param1,
                            reg_param2, xform_params=None):
                            
    # D0 : dimension of base latent vectors
    # D  : dimension of transformed latent vectors
    
    rng = np.random.RandomState()

    # Initialize latent vectors
    offense0_vals = np.asarray(rng.uniform(
        low  = -np.sqrt(6./(num_teams+D0)),
        high =  np.sqrt(6./(num_teams+D0)),
        size = (num_teams, D0)), dtype=theano.config.floatX)
    defense0_vals = np.asarray(rng.uniform(
        low  = -np.sqrt(6./(num_teams+D0)),
        high =  np.sqrt(6./(num_teams+D0)),
        size = (num_teams, D0)), dtype=theano.config.floatX)

    # Initialize transformation weights.
    # Different transform for {offense, defense} x {home, away, tourney}
    if xform_params is None:
        oxformW1_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(6./(D0+D)),
            high =  np.sqrt(6./(D0+D)),
            size = (3,D0,H)), dtype=theano.config.floatX)
        oxformb1_vals = np.zeros((D,), dtype=theano.config.floatX)
        dxformW1_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(6./(D0+D)),
            high =  np.sqrt(6./(D0+D)),
            size = (3,D0,H)), dtype=theano.config.floatX)
        dxformb1_vals = np.zeros((D,), dtype=theano.config.floatX)
        oxformW2_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(6./(D0+D)),
            high =  np.sqrt(6./(D0+D)),
            size = (3,H,D)), dtype=theano.config.floatX)
        oxformb2_vals = np.zeros((D,), dtype=theano.config.floatX)
        dxformW2_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(6./(D0+D)),
            high =  np.sqrt(6./(D0+D)),
            size = (3,H,D)), dtype=theano.config.floatX)
        dxformb2_vals = np.zeros((D,), dtype=theano.config.floatX)
        paceW1_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(1./(4*D0+1)),
            high =  np.sqrt(1./(4*D0+1)),
            size = (4*D,Hp)), dtype=theano.config.floatX)
        paceb1_vals = np.zeros((Hp,), dtype=theano.config.floatX)
        paceW2_vals = np.asarray(rng.uniform(
            low  = -np.sqrt(1./(4*D0+1)),
            high =  np.sqrt(1./(4*D0+1)),
            size = (Hp,1)), dtype=theano.config.floatX)
        paceb2_vals = np.zeros((1,), dtype=theano.config.floatX)
        mean_score_val = 70*np.ones(1, dtype=theano.config.floatX)

        # Create theano variables
        oxformW1 = theano.shared(value=oxformW1_vals, name="oxformW1")
        oxformb1 = theano.shared(value=oxformb1_vals, name="oxformb1")
        dxformW1 = theano.shared(value=dxformW1_vals, name="dxformW1")
        dxformb1 = theano.shared(value=dxformb1_vals, name="dxformb1")
        oxformW2 = theano.shared(value=oxformW2_vals, name="oxformW2")
        oxformb2 = theano.shared(value=oxformb2_vals, name="oxformb2")
        dxformW2 = theano.shared(value=dxformW2_vals, name="dxformW2")
        dxformb2 = theano.shared(value=dxformb2_vals, name="dxformb2")
        paceW1 = theano.shared(value=paceW1_vals, name="paceW1")
        paceb1 = theano.shared(value=paceb1_vals, name="paceb1")
        paceW2 = theano.shared(value=paceW2_vals, name="paceW2")
        paceb2 = theano.shared(value=paceb2_vals, name="paceb2")
        mean_score = theano.shared(value=mean_score_val, name="mean_score")
    else:
        # This allows sharing of transform parameters across seasons
        oxformW1, oxformb1, dxformW1, dxformb1, \
                 oxformW2, oxformb2, dxformW2, dxformb2, \
                 paceW1, paceb1, paceW2, paceb2, mean_score \
                 = xform_params

        
    offenses0 = theano.shared(value=offense0_vals, name="offenses0")
    defenses0 = theano.shared(value=defense0_vals, name="defenses0")

    # inputs
    SUPPORT_BATCH_LEARNING = False  # theano indexing is not cooperating
    if SUPPORT_BATCH_LEARNING:
        team1_ids    = T.ivector("team1_ids")
        team1_locs   = T.ivector("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.ivector("team2_ids")
        team2_locs   = T.ivector("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dvector("team1_scores")
        team2_scores = T.dvector("team2_scores")
    else:   # only support stochastic gradient training
        team1_ids    = T.iscalar("team1_ids")
        team1_locs   = T.iscalar("team1_locs")     # 0:home, 1:away, 2:tourney
        team2_ids    = T.iscalar("team2_ids")
        team2_locs   = T.iscalar("team2_locs")     # 0:home, 1:away, 2:tourney
        team1_scores = T.dscalar("team1_scores")
        team2_scores = T.dscalar("team2_scores")

    # learning parameters
    learning_rate = T.scalar("learning_rate")

    # select appropriate latent vectors
    team1_offenses0 = offenses0[team1_ids,:]
    team1_defenses0 = defenses0[team1_ids,:]
    team2_offenses0 = offenses0[team2_ids,:]
    team2_defenses0 = defenses0[team2_ids,:]

    # apply location-specific transformations
    # 0: home; 1:away; 2:tourney
    team1_oxformW1s = oxformW1[team1_locs]
    team1_oxformb1s = oxformb1[team1_locs]
    team2_oxformW1s = oxformW1[team2_locs]
    team2_oxformb1s = oxformb1[team2_locs]
    team1_dxformW1s = dxformW1[team1_locs]
    team1_dxformb1s = dxformb1[team1_locs]
    team2_dxformW1s = dxformW1[team2_locs]
    team2_dxformb1s = dxformb1[team2_locs]
    
    team1_oxformW2s = oxformW2[team1_locs]
    team1_oxformb2s = oxformb2[team1_locs]
    team2_oxformW2s = oxformW2[team2_locs]
    team2_oxformb2s = oxformb2[team2_locs]
    team1_dxformW2s = dxformW2[team1_locs]
    team1_dxformb2s = dxformb2[team1_locs]
    team2_dxformW2s = dxformW2[team2_locs]
    team2_dxformb2s = dxformb2[team2_locs]
    
    # transformed offenses and defenses for each game
    team1_offensesH = T.dot(team1_offenses0, team1_oxformW1s) + team1_oxformb1s
    team1_defensesH = T.dot(team1_defenses0, team1_dxformW1s) + team1_dxformb1s
    team2_offensesH = T.dot(team2_offenses0, team2_oxformW1s) + team2_oxformb1s
    team2_defensesH = T.dot(team2_defenses0, team2_dxformW1s) + team2_dxformb1s

    team1_offenses = T.dot(team1_offensesH, team1_oxformW2s) + team1_oxformb2s
    team1_defenses = T.dot(team1_defensesH, team1_dxformW2s) + team1_dxformb2s
    team2_offenses = T.dot(team2_offensesH, team2_oxformW2s) + team2_oxformb2s
    team2_defenses = T.dot(team2_defensesH, team2_dxformW2s) + team2_dxformb2s

    if SUPPORT_BATCH_LEARNING:
        team1_pred_score0 = T.sum(team1_offenses * team2_defenses, axis=1)
        team2_pred_score0 = T.sum(team2_offenses * team1_defenses, axis=1)
    else:
        team1_pred_score0 = T.sum(team1_offenses * team2_defenses)
        team2_pred_score0 = T.sum(team2_offenses * team1_defenses)

    all_o_and_d_12 = T.concatenate([team1_offenses, team1_defenses,
                                    team2_offenses, team2_defenses])
    all_o_and_d_21 = T.concatenate([team2_offenses, team2_defenses,
                                    team1_offenses, team1_defenses])

    paceH = T.nnet.sigmoid(T.dot(all_o_and_d_12, paceW1) \
                           + T.dot(all_o_and_d_21, paceW1) + paceb1)
    pace = .5 + T.nnet.sigmoid(T.dot(paceH, paceW2) + paceb2)

    team1_pred_score = (team1_pred_score0 + mean_score) * pace
    team2_pred_score = (team2_pred_score0 + mean_score) * pace
    
    # regularization terms
    reg1 = T.mean(T.sqr(offenses0)) + T.mean(T.sqr(defenses0))
    reg2 = T.mean(T.sqr(oxformW1)) + T.mean(T.sqr(dxformW1)) \
           + T.mean(T.sqr(oxformW2)) + T.mean(T.sqr(dxformW2)) \
           + T.mean(T.sqr(paceW1)) + T.mean(T.sqr(paceW2))

    # learning objective
    obj = T.mean(T.sqr(team1_pred_score - team1_scores)) + \
          T.mean(T.sqr(team2_pred_score - team2_scores)) + \
          reg_param1 * reg1 + reg_param2 * reg2

    # Define updates
    params = [offenses0, defenses0,
              oxformW1, oxformb1, dxformW1, dxformb1,
              oxformW2, oxformb2, dxformW2, dxformb2,
              paceW1, paceb1, paceW2, paceb2, mean_score]

    grads  = []
    for p in params:
        g = T.grad(obj, p)
        grads.append(g)

    updates = {}
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    # Create and return theano functions
    out_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs],
                             outputs=[team1_pred_score, team2_pred_score])

    train_fn = theano.function([team1_ids, team1_locs, team2_ids, team2_locs,
                                team1_scores, team2_scores, learning_rate],
                               outputs=obj,
                               updates=updates)

    return out_fn, train_fn, params
