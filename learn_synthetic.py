from model import *


if __name__ == "__main__":
    N  = 300   # num teams
    # only used for matrix factorization algorithms
    D  = 10    # output latent vector dimension
    Hp = 5     # hidden units for pacing component
    
    # only used for matrix factorization algorithms
    D0 = 30    # base latent vector dimension
    H = 20     # num hidden units for transformation networks

    reg_param1 = .001
    reg_param2 = .001

    NUM_ITS = 500

    MODEL = "pmf_with_pace"
    if MODEL == "simplest":
        BASE_LEARNING_RATE = .005
        make_model_fn = make_simplest_learning_functions

    elif MODEL == "pmf":  # probabilistic matrix factorization
        BASE_LEARNING_RATE = .005
        make_model_fn = make_vanilla_pmf_functions

    elif MODEL == "pmf_with_pace":
        BASE_LEARNING_RATE = .001
        make_model_fn = make_pmf_plus_pace_functions

    elif MODEL == "full":
        BASE_LEARNING_RATE = .0001
        make_model_fn = make_learning_functions

    else:
        assert False  # unsupported model

    out_fn, train_fn, params = \
            make_model_fn(N, D0, H, D, Hp, reg_param1, reg_param2)

    # make synthetic data
    G = 500   # num games
    team1_ids = np.random.randint(N,size=(G))
    team2_ids = np.random.randint(N,size=(G))
    team1_locs = np.random.randint(3, size=(G))
    team2_locs = np.random.randint(3, size=(G))
    team1_scores = np.random.randint(50, high=70, size=(G))
    team2_scores = np.random.randint(50, high=70, size=(G))

    print team1_ids
    print team1_locs
    
    for t in range(NUM_ITS):
        obj = 0
        learning_rate = BASE_LEARNING_RATE / (1.0 + np.sqrt(t))
        for g in range(G):
            obj_g = train_fn(team1_ids[g], team1_locs[g], team2_ids[g], team2_locs[g],
                             team1_scores[g], team2_scores[g], learning_rate)
            obj += obj_g

            pred_team1, pred_team2 = out_fn(team1_ids[g], team1_locs[g],
                                            team2_ids[g], team2_locs[g])

            if t % 10 == 0 and g % 50 == 0:
                print "%s-%s vs %s-%s" % (pred_team1, pred_team2,
                                          team1_scores[g], team2_scores[g])
        
        print "%s\t%s" % (t, obj)
