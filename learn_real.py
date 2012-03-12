import sys
from model import *
from march_madness_data import *
import matplotlib.pylab as plt


def train(data, split):

    N  = len(data.team_codes)  # num teams
    # only used for matrix factorization algorithms
    D  = 2    # output latent vector dimension
    Hp = 5     # hidden units for pacing component
    
    # only used for matrix factorization algorithms
    D0 = 10    # base latent vector dimension
    H = 20     # num hidden units for transformation networks

    reg_param1 = 0.0
    reg_param2 = 0

    NUM_ITS = 20

    SEASONS_ARE_INDEPENDENT = True
    seasons = ["2006-2007", "2007-2008", "2008-2009", "2009-2010", "2010-2011"]
    train_on_season, train_on_tourney, test_on_tourney = {}, {}, {}
    
    for s, season in enumerate(seasons):
        if SEASONS_ARE_INDEPENDENT and s != split:
            train_on_season[season] = False
            train_on_tourney[season] = False
            test_on_tourney[season] = False

        elif SEASONS_ARE_INDEPENDENT:
            train_on_season[season] = True
            train_on_tourney[season] = False
            test_on_tourney[season] = True

        else:
            train_on_season[season] = True
            if s == split:
                train_on_tourney[season] = False
                test_on_tourney[season] = True
            else:
                train_on_tourney[season] = True
                test_on_tourney[season] = False
        
        print season, train_on_season[season], train_on_tourney[season], test_on_tourney[season]

    MODEL = "pmf"
    if MODEL == "simplest":
        BASE_LEARNING_RATE = .005
        make_model_fn = make_simplest_learning_functions

    elif MODEL == "pmf":  # probabilistic matrix factorization
        BASE_LEARNING_RATE = .0025
        make_model_fn = make_vanilla_pmf_functions

    elif MODEL == "pmf_with_pace":
        BASE_LEARNING_RATE = .001
        make_model_fn = make_pmf_plus_pace_functions

    elif MODEL == "full":
        BASE_LEARNING_RATE = .0001
        make_model_fn = make_learning_functions

    else:
        assert False  # unsupported model

    params = {}
    train_fns = {}
    out_fns = {}
    s_ctr = 0
    for s, season in enumerate(seasons):
        # no reason to instantiate memory if we're not training or testing
        if not train_on_season[season] and \
           not train_on_tourney[season] and \
           not test_on_tourney[season]:  continue

        print "Making weights for %s" % season

        if s_ctr == 0:
            out_fns[season], train_fns[season], params[season] = \
                        make_model_fn(N, D0, H, D, Hp, reg_param1, reg_param2)
            season0 = season
        else:
            out_fns[season], train_fns[season], params[season] = \
                        make_model_fn(N, D0, H, D, Hp, reg_param1, reg_param2,
                                      xform_params=params[season0][2:])
        s_ctr += 1

    training_game_chunks = []
    testing_game_chunks = []
    for s, season in enumerate(seasons):
        season_games = data.game_results_by_season[season]
        tourney_games = data.tourney_game_results_by_season[season]

        if train_on_season[season]:
            training_game_chunks.append([season_games, season, "season"])

        if train_on_tourney[season]:
            training_game_chunks.append([tourney_games, season, "tourney"])

        if test_on_tourney[season]:
            testing_game_chunks.append([tourney_games, season, "tourney"])


    train_objs, sq_valid_errs, zero_one_valid_errs = [], [], []
    for t in range(NUM_ITS):
        obj = 0
        learning_rate = np.float32(BASE_LEARNING_RATE / (1.0 + np.sqrt(t)))
        ngames = 0
        for game_chunk in training_game_chunks:
            games, season, description = game_chunk

            print "Train: %s %s" % (s, description)

            train_fn = train_fns[season]
            out_fn = out_fns[season]

            # stochastic training updates
            for g, game in enumerate(games):
                team1_id, team2_id = game[1], game[2]
                team1_score, team2_score = game[3], game[4]
                if description == "season":
                    team1_loc, team2_loc = LOCATION_HOME, LOCATION_AWAY
                else:
                    team1_loc, team2_loc = LOCATION_TOURNEY, LOCATION_TOURNEY                    
                    
                
                if g % 1000 == 0:
                    pred_team1, pred_team2 = out_fn(team1_id, team1_loc,
                                                    team2_id, team2_loc)
                    print "\t%s\t %s-%s vs %s-%s" % (description, pred_team1, pred_team2,
                                                    team1_score, team2_score)


                obj_g = train_fn(team1_id, team2_loc, team2_id, team2_loc,
                                 team1_score, team2_score, learning_rate)
                obj += obj_g
                ngames += 1.0
        train_objs.append(obj/ngames)
        
        # compute test/validation score
        sq_valid_err = 0
        zero_one_valid_err = 0
        ngames = 0
        for game_chunk in testing_game_chunks:
            games, season, description = game_chunk
            out_fn = out_fns[season]
            
            print "Valid: %s %s" % (s, description)            
            for g, game in enumerate(games):
                team1_id, team2_id = game[1], game[2]
                team1_score, team2_score = game[3], game[4]

                pred_team1, pred_team2 = out_fn(team1_id, team1_loc,
                                                team2_id, team2_loc)

                sq_loss = validation_loss(pred_team1, pred_team2, team1_score, team2_score,
                                          method="sqerr")
                zero_one_loss = validation_loss(pred_team1, pred_team2, team1_score, team2_score,
                                                method="zero-one")
                sq_valid_err += sq_loss
                zero_one_valid_err += zero_one_loss
                print "\tTourney %s-%s vs %s-%s (losses=%s %s)" % (pred_team1, pred_team2,
                                                                   team1_score, team2_score,
                                                                   sq_loss, zero_one_loss)
                ngames += 1.0

        print "%s\tValidation err\t%s\t%s" % (t, sq_valid_err, zero_one_valid_err)
        sq_valid_errs.append(sq_valid_err/ngames)
        zero_one_valid_errs.append(zero_one_valid_err)

    return train_objs, sq_valid_errs, zero_one_valid_errs


if __name__ == "__main__":

    try:
        splits = [int(sys.argv[1])]
    except:
        splits = [0, 1, 2, 3, 4]

    data = MarchMadnessData()

    num_splits = len(splits)
    num_cols = 3
    for s, split in enumerate(splits):
        train_objs, sq_valid_errs, zero_one_valid_errs = train(data, split)
    
        plt.subplot(num_splits,num_cols,num_cols*s+1)
        plt.plot(train_objs, 'r-')
        plt.legend(["Train"])
        plt.ylabel("Season %s" % split)
        plt.xlabel("Iteration")
        
        plt.subplot(num_splits,num_cols,num_cols*s+2)
        plt.plot(sq_valid_errs, 'b-')
        plt.legend(["Valid (sq)"])
        plt.xlabel("Iteration")
        
        plt.subplot(num_splits,num_cols,num_cols*s+3)
        plt.plot(zero_one_valid_errs, 'k-')
        plt.legend(["Valid (0-1)"])
        plt.xlabel("Iteration")
        
    plt.show()

        
