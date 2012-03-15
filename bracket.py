import pickle
import numpy
import re

class Bracket:

    def __init__(self):
        self.num_rounds = 6
        self.round = {}
        self.bracket = {}
        for r in range(self.num_rounds):
            self.round[r] = []  #(64 / 2**r) * [None]
            self.bracket[r] = 2**(self.num_rounds-r) * [None]


    def load_starting_configuration(self, bracket_filename):
        f = open(bracket_filename, 'r') # e.g., '2009_bracket.txt'

        non_white_re = re.compile(r'\w+')

        self.starting_teams = []
        for line in f:
            line = line.strip()
            m = non_white_re.search(line)
            if m is not None:
                self.starting_teams.append(line)    


    def predict_tournament(self, out_fn, team_code_to_id):
        self.bracket[0] = []
        for team in self.starting_teams:
            self.bracket[0].append(team)

        print "initial b[0]"
        print self.bracket[0]

        for r in range(self.num_rounds):
            self.bracket[r+1] = []
            for g in range(len(self.bracket[r]) / 2):
                t1 = self.bracket[r][2*g]
                t2 = self.bracket[r][2*g+1]

                t1_id = team_code_to_id[t1]
                t2_id = team_code_to_id[t2]

                t1_score, t2_score = out_fn(t1_id, 2, t2_id, 2)  # 2 means tournament game

                if t1_score > t2_score:  self.bracket[r+1].append(t1)
                else:                    self.bracket[r+1].append(t2)

        self.print_full()
        

    def opponent_in_round(self, team_id, r):
        """ Find who team <team> played in round <r> """

        for game in self.round[r]:
            date, home_id, away_id, home_score, away_score = game
            if   home_id == team_id: return away_id
            elif away_id == team_id: return home_id

        assert False, "Did not find opponent for %s in %s" % (team_id, self.round[r])
        return None


    def make_bracket_structure(self, team_codes=None):
        """ Reorder games within rounds to make bracket structure
        sensible -- that is, the winner of the current round's i^{th} game
        should play in next round's floor(i / 2) game.
        """

        # start at last round and work backwards.  assign previous
        # game of "home" team to the lower game id, and "away" team
        # to the higher game id.
        date, home_id, away_id, home_score, away_score = self.round[5][0]
        winner_id = home_id if home_score > away_score else away_id
        loser_id = away_id if home_score > away_score else home_id

        assert winner_id is not None
        assert loser_id is not None
        
        self.bracket[5][0] = winner_id
        self.bracket[5][1] = loser_id

        for r in reversed(range(self.num_rounds)):
            num_games = 2**(self.num_rounds - r)

            if r == 0:  continue

            for i in range(num_games):
                # place winners
                winner_id = self.bracket[r][i]
                self.bracket[r-1][i*2] = winner_id

                loser_id = self.opponent_in_round(winner_id, r-1)
                self.bracket[r-1][i*2+1] = loser_id

                assert winner_id is not None
                assert loser_id is not None

        if team_codes is not None:
            # replace ids with team codes
            for r in range(self.num_rounds):
                for i, team_id in enumerate(self.bracket[r]):
                    self.bracket[r][i] = team_codes[team_id]
        

    def print_full(self):
        line_sep = "---"
        blank_sep = "   "
        for i, team_code in enumerate(self.bracket[0]):
            line = ""
            first_blank = True
            for r in range(self.num_rounds):
                if team_code in self.bracket[r]:
                    line += "%s%s" % ("" if r == 0 else line_sep, team_code)
                else:
                    ii = i
                    r1 = r+0
                    r2 = r+1

                    if first_blank:
                        line += "%s" % blank_sep
                        first_blank = False
                        
                    if ii % (2**r2) <= ii % (2**r1): line += "  |%s"  % blank_sep
                    else: line += "   %s" % blank_sep
            print line

            if i == len(self.bracket[0]) - 1:  break
            
            for n in [4, 16]:  # print extra space at these intervals
                if (i+1) % n == 0:
                    alt_line = "   " + line[3:]
                    print alt_line
        print


    def simulate_game(self, team1_code, team2_code):
        # TODO
        pass
    

if __name__ == '__main__':
    from march_madness_data import *

    data = MarchMadnessData()

    for season in data.brackets:
        print season
        data.brackets[season].print_full()
        print


    
