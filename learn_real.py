from model import *
from march_madness_data import *


if __name__ == "__main__":
    
    data = MarchMadnessData()
    
    N  = len(data.team_codes)  # num teams
    # only used for matrix factorization algorithms
    D  = 10    # output latent vector dimension
    Hp = 5     # hidden units for pacing component
    
    # only used for matrix factorization algorithms
    D0 = 30    # base latent vector dimension
    H = 20     # num hidden units for transformation networks

    reg_param1 = .0
    reg_param2 = .0

    NUM_ITS = 500
    
    split = 1
    train_seasons = ["2006-2007", "2007-2008", "2008-2009", "2009-2010"]

    # take one train season away, and use it as validation
    # TODO: not implemented yet
    validation_season = train_seasons.pop(split)
    validation_seasons = [validation_season]

    MODEL = "pmf"
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

    params = {}
    train_fns = {}
    out_fns = {}
    for s, season in enumerate(train_seasons):
        if s == 0:
            out_fns[0], train_fns[0], params[0] = \
                        make_model_fn(N, D0, H, D, Hp, reg_param1, reg_param2)
        else:
            out_fns[s], train_fns[s], params[s] = \
                        make_model_fn(N, D0, H, D, Hp, reg_param1, reg_param2,
                                      xform_params=params[0][2:])

    for t in range(NUM_ITS):
        obj = 0
        learning_rate = BASE_LEARNING_RATE / (1.0 + np.sqrt(t))
        for season in train_seasons:
            print t, season
            train_fn = train_fns[s]
            out_fn = out_fns[s]

            # training regular season updates
            games = data.game_results_by_season[season]
            for g, game in enumerate(games):
                team1_id, team2_id = game[1], game[2]
                team1_score, team2_score = game[3], game[4]
                team1_loc, team2_loc = LOCATION_HOME, LOCATION_AWAY
                
                if g % 1000 == 0:
                    pred_team1, pred_team2 = out_fn(team1_id, team1_loc,
                                                    team2_id, team2_loc)
                    print "\tSeason %s-%s vs %s-%s" % (pred_team1, pred_team2,
                                                       team1_score, team2_score)

                obj_g = train_fn(team1_id, team2_loc, team2_id, team2_loc,
                                 team1_score, team2_score, learning_rate)
                obj += obj_g
                
            # training tourney updates -- TODO: maybe use one of these
            # as validation?
            games = data.tourney_game_results_by_season[season]
            for g, game in enumerate(games):
                team1_id, team2_id = game[1], game[2]
                team1_score, team2_score = game[3], game[4]
                team1_loc, team2_loc = LOCATION_HOME, LOCATION_AWAY
                
                if g % 10 == 0:
                    pred_team1, pred_team2 = out_fn(team1_id, team1_loc,
                                                    team2_id, team2_loc)
                    print "\tTourney %s-%s vs %s-%s" % (pred_team1, pred_team2,
                                                        team1_score, team2_score)

                obj_g = train_fn(team1_id, team2_loc, team2_id, team2_loc,
                                 team1_score, team2_score, learning_rate)
                obj += obj_g

            # TODO: test regular season updates
        
        print "%s\t%s" % (t, obj)
