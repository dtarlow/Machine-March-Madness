DATA_PATH = "./data"
FULL_DATA_PATH = "%s/Data_20110823" % DATA_PATH
GAMES_CSV_LOC = "%s/Games.tsv" % FULL_DATA_PATH
PLAYERS_CSV_LOC = "%s/Players.tsv" % FULL_DATA_PATH
SIMPLE_DATA_LOC = "%s/GameResults_20110311.tsv" % DATA_PATH
TEAM_MAPPING_LOC = "%s/YahooTeamCodeMapping.csv" % DATA_PATH

PAST_WINNERS = {
    "2006-2007" : "fak",
    "2007-2008" : "kaa",
    "2008-2009" : "nav",
    "2009-2010" : "dau",
    "2010-2011" : "cbp",
    "2011-2012" : None
    }


import numpy as np
from bracket import Bracket


class MarchMadnessData:

    def __init__(self):
        self.load_team_mappings()

        self.load_simple_data()

        print "After loading simple data"
        self.print_season_infos()
        
        self.determine_tourney_games_and_make_brackets()

        print "After removing tournament games"
        self.split_postseason_games_from_seasons()

        self.print_season_infos()

        USE_FULL_DATA = False
        if USE_FULL_DATA:
            # Warning: not fully implemented or checked for accuracy
            self.load_full_data()


    def load_team_mappings(self):
        f = open(TEAM_MAPPING_LOC, 'r')
        self.team_code_to_id = {}
        self.team_codes = []
        for team_id, line in enumerate(f):
            team_code, team_name = line.rstrip().split(",")

            self.team_codes.append(team_code)
            self.team_code_to_id[team_code] = team_id


    def load_simple_data(self):
        self.game_results_by_season = {}
        
        f = open(SIMPLE_DATA_LOC, 'r')
        num_skipped = 0
        for line in f:
            date, home_code, away_code, home_score, away_score, home_won = \
                  line.rstrip().split("\t")

            year, month, day = date[:4], date[4:6], date[6:]
            year, month, day = int(year), int(month), int(day)

            if month < 6:  season = "%s-%s" % (year-1,year)
            else:          season = "%s-%s" % (year,year+1)

            # We only have partial data for 2005-2006.  Skip it.
            if season == "2005-2006":  continue

            # We don't have aggregate data for the 2010-2011 tournament
            # at the moment.  Skip as well.
            #
            # Update: we have it now
            #if season == "2010-2011":  continue

            if season not in self.game_results_by_season:
                self.game_results_by_season[season] = []

            if home_code == "UNK" or away_code == "UNK":
                # HACK!  my script breaks if an UNK occurs in the tournament, so
                # we're making up an opponent for wisconsin there.
                if home_code == "wbg" and away_code == "UNK" and home_score == "76":
                    away_code="tan"
                    print "Fixed opponent for Wisconsin game on %s-%s-%s" % (year, month, day)
                else:
                    num_skipped += 1
                    continue

            home_score = int(home_score)
            away_score = int(away_score)
            home_won = bool(home_won)

            home_id = self.team_code_to_id[home_code]

            away_id = self.team_code_to_id[away_code]

            record = [(year, month, day), \
                      home_id, away_id, home_score, away_score]

            self.game_results_by_season[season].append(record)

            
        print "Skipped %s entries due to UNK" % (num_skipped)


    def load_full_data(self):

        # TODO/WARNING: finish this up.  right now it loads the files, but
        # doesn't do anything interesting with the data, and I have not
        # verified that these results match up against the aggregate data.
        
        self.team_scores = {}

        f = open(GAMES_CSV_LOC, 'r')
        for line in f:
            game_id, date, home_code, away_code = line.rstrip().split("\t")

            if away_code == "UNK" or home_code == "UNK":  continue

            self.team_scores[game_id] = {}
            self.team_scores[game_id][home_code] = 0
            self.team_scores[game_id][away_code] = 0

        f.close()

        f = open(PLAYERS_CSV_LOC, 'r')
        num_skipped = 0
        for line in f:
            name, game_id, team_code, minutes, fg_made, fg_att, \
                  three_made, three_att, free_made, free_att, \
                  off_reb, def_reb, assists, turnovers, steals, \
                  blocks, fouls = line.rstrip().split("\t")

            if game_id not in self.team_scores:
                num_skipped += 1
                continue

            fg_made, three_made, free_made = int(fg_made), int(three_made), int(free_made)

            player_pts = 2*fg_made + three_made + free_made
            
            self.team_scores[game_id][team_code] += player_pts

        f.close()

        print self.team_scores
        print "Skipped %s entries" % num_skipped


    def print_season_infos(self):
        
        print "\n".join(["%s: %s games" % (s, len(self.game_results_by_season[s])) \
                         for s in sorted(self.game_results_by_season)])


    def determine_tourney_games_and_make_brackets(self):
        """
        We don't have data saying which games were a part of the
        tournament, so we have to create it programmatically.
        
        Some assumptions:
            - all tournament games occur in March or April
            - the last game played by the winning team was the
              championship game
            - all round i games finish before any round i+1 games start

        Note we can't just take the last 63 games, because there
        are e.g., NIT tournament games mixed in.
        """

        self.brackets = {}
        self.tournament_starts = {}
        for season in PAST_WINNERS:

            if season == "2011-2012":  continue
            
            if season not in self.game_results_by_season:  continue

            bracket = Bracket()
            
            dummy, tourney_year = season.split("-")
            games = []
            for i, game in enumerate(self.game_results_by_season[season]):

                (date, home_id, away_id, home_score, away_score) = game

                year, month, day = date
            
                if not (month == 3 or month == 4):  continue

                home_code = self.team_codes[home_id]
                away_code = self.team_codes[away_id]

                games.append((month*100+day, home_code, away_code, game))

            surviving_teams = [PAST_WINNERS[season]]
            bracket.round[6] = surviving_teams
            for game in reversed(sorted(games)):
                num_surviving = len(surviving_teams)

                home_code, away_code = game[1], game[2]
                if home_code in surviving_teams:
                    surviving_teams.append(away_code)
                elif away_code in surviving_teams:    
                    surviving_teams.append(home_code)
                else:
                    continue

                r = 5 - int(np.floor(np.log2(num_surviving)))
                if r < 0:  break
                bracket.round[r].append(game[3])

                if season not in self.tournament_starts or \
                       game[0] < self.tournament_starts[season]:
                    self.tournament_starts[season] = game[0]

            bracket.make_bracket_structure(team_codes=self.team_codes)

            self.brackets[season] = bracket
            

    def split_postseason_games_from_seasons(self):
        """ Don't include post-season games as a part of the regular season. """

        self.tourney_game_results_by_season = {}
        for season in self.game_results_by_season:
            season_copy = []
            tourney_games = []
            for i, game in enumerate(self.game_results_by_season[season]):

                (date, home_id, away_id, home_score, away_score) = game
                
                year, month, day = date
                
                if season not in self.tournament_starts or month > 6 or \
                       month*100+day < self.tournament_starts[season]:
                    season_copy.append(game)
                else:
                    tourney_games.append(game)

            self.game_results_by_season[season] = season_copy
            self.tourney_game_results_by_season[season] = tourney_games
                    

if __name__ == "__main__":

    data = MarchMadnessData()
