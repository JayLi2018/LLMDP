import numpy as np
import json
import re
import logging
import LLMDP.logconfig
import logging
import random
import pdb

logger = logging.getLogger(__name__)

def extract_response(content):
    """
    Extract label, keywords explanations and regular expressions from GPT response
    """
    label_match = re.search("LABEL:\s*\d+$", content, flags=re.M)
    if label_match:
        st, ed = label_match.span()
        label = int(label_match.string[st + 6:ed])
    else:
        label = None

    regex_match = re.search("REGEX:.*$", content, flags=re.M)
    if regex_match:
        st, ed = regex_match.span()
        regex_list = []
        for x in regex_match.string[st+6:ed].split('[SEP]'):
            regex = x.strip(" '\"\n")
            if regex.lower() != "none":
                regex_list.append(regex)

    else:
        regex_list = None

    keyword_match = re.search("KEYWORDS:.*$", content, flags=re.M)
    if keyword_match:
        st, ed = keyword_match.span()
        keyword_list = [x.strip() for x in keyword_match.string[st + 9:ed].split(',')]

    else:
        keyword_list = None

    explanation_match = re.search("Explanation:.*$", content, flags=re.M)
    if explanation_match:
        st, ed = explanation_match.span()
        explanation = explanation_match.string[st+12:ed]
    else:
        explanation = None

    response_dict = {
        "label": label,
        "keyword_list": keyword_list,
        "regex_list": regex_list,
        "explanation": explanation
    }
    return response_dict


def create_user_prompt(example_prompt, dataset_name, dataset, query_idx, user_provide_instance_label=False):
    """
    Create the user prompt with few shot in context learning
    """
    if dataset_name in ["cdr", "chemprot", "semeval", "spouse"]:
        text = dataset.examples[query_idx]["text"]
        entity1 = dataset.examples[query_idx]["entity1"]
        entity2 = dataset.examples[query_idx]["entity2"]
        if dataset_name == "cdr":
            user_prompt = "{} User: {}. Does {} cause{}?\n Response: ".format(example_prompt, text, entity1, entity2)
        elif dataset_name == "spouse":
            user_prompt = "{} User: {}. Are {} and {} spouses?\n Response: ".format(example_prompt, text, entity1, entity2)
        else:
            user_prompt = "{} User: {}. What is the relationship between {} and {}?\n Response: ".format(
                            example_prompt, text, entity1, entity2)
        return user_prompt
    else:
        if(not user_provide_instance_label):
            text = dataset.examples[query_idx]["text"]
            label = dataset.labels[query_idx]
            user_prompt = "{} User: {}\n Response: ".format(example_prompt, text)
            return user_prompt, label
        else:
            text = dataset.examples[query_idx]["text"]
            label = dataset.labels[query_idx]
            # pdb.set_trace()
            user_prompt = "{} User: {}\n The label for this text:\n {} \n Response: ".format(example_prompt, text, label)
            return user_prompt, label


def create_cot_user_prompt(example_prompt, dataset_name, dataset, query_idx):
    """
    Create the user prompt that ask for explanation, keywords or regex given text and label
    """
    if dataset_name in ["cdr", "chemprot", "semeval", "spouse"]:
        text = dataset.examples[query_idx]["text"]
        label = dataset.labels[query_idx]
        entity1 = dataset.examples[query_idx]["entity1"]
        entity2 = dataset.examples[query_idx]["entity2"]
        if dataset_name == "cdr":
            user_prompt = "{} User: {}. Does {} cause{}?\nLabel: {}\n Response: ".format(example_prompt, text, entity1, entity2, label)
        else:
            user_prompt = "{} User: {}. What is the relationship between {} and {}?\nLabel:{}\nResponse: ".format(
                            example_prompt, text, entity1, entity2, label)
    else:
        text = dataset.examples[query_idx]["text"]
        label = dataset.labels[query_idx]
        user_prompt = "{} User: {}\nLabel:{}\nResponse: ".format(example_prompt, text, label)

    return user_prompt


def build_example(dataset_name, dataset, query_idx, response_dict):
    """
    Build an in-context example from response
    """
    if dataset_name in ["cdr", "chemprot", "semeval", "spouse"]:
        text = dataset.examples[query_idx]["text"]
        entity1 = dataset.examples[query_idx]["entity1"]
        entity2 = dataset.examples[query_idx]["entity2"]
        label = dataset.labels[query_idx]
        if dataset_name == "cdr":
            user_prompt = "User: {}. Does {} cause{}?\n".format(text, entity1,entity2)
        else:
            user_prompt = "User: {}. What is the relationship between {} and {}?\n".format(text, entity1, entity2)

    else:
        text = dataset.examples[query_idx]["text"]
        label = dataset.labels[query_idx]
        user_prompt = "User: {}\n".format(text)

    response = "Response:\n"
    if response_dict["explanation"] is not None:
        response += "Explanation:{}\n".format(response_dict["explanation"])

    if response_dict["regex_list"] is not None:
        response += "REGEX:{}\n".format("[SEP]".join(response_dict["regex_list"]))

    if response_dict["keyword_list"] is not None:
        response += "KEYWORDS:{}\n".format(",".join(response_dict["keyword_list"]))

    response += "LABEL:{}\n".format(    )
    return user_prompt+response


def create_cot_prompt(dataset_name, dataset, example_per_class=1, **kwargs):
    """
    Create prompt that ask LLM to generate chain-of-thought and/or keywords automatically given instance and label
    """
    logger.warning("creating few shot example")
    if dataset_name == "youtube":
        task = "spam classification"
        task_info = "In each iteration, the user will provide a comment for a video and a label indicating whether the comment is a spam. "
        class_info = "0 for non-spam, 1 for spam"
    elif dataset_name == "sms":
        task = "spam classification"
        task_info = "In each iteration, the user will provide a text message and a label indicating whether the message is a spam. "
        class_info = "0 for non-spam, 1 for spam"
    elif dataset_name == "imdb":
        task = "sentiment analysis"
        task_info = "In each iteration, the user will provide a movie review and a label indicating whether the review is positive or negative."
        class_info = "0 for negative, 1 for positive"
    elif dataset_name == "yelp":
        task = "sentiment analysis"
        task_info = "In each iteration, the user will provide a restaurant review and a label indicating whether the review is positive or negative."
        class_info = "0 for negative, 1 for positive"
    elif dataset_name == "chemprot":
        task = "biomedical relation extraction"
        task_info = "In each iteration, the user will provide a biomedical statement, followed by a question asking the relationship between two chemicals occured in that statement." \
                    "Then the user will provide a label indicating the relationship between the two chemicals based on the statement."
        class_info = "0 for chemical B (or A) is part of chemical A (or B), 1 for chemical B (or A) is the regulator of chemical A (or B). " \
                     "2 for chemical B (or A) is the upregulator of chemical A (or B). 3 for chemical B (or A) is the downregulator of chemical A (or B)." \
                     "4 for chemical B (or A) is the agnoist of chemical A (or B). 5 for chemical B (or A) is the antagonist of chemical A (or B)." \
                     "6 for chemical B (or A) is the modulator of chemical A (or B). 7 for chemical B (or A) is the cofactor of chemical A (or B)." \
                     "8 for chemical B (or A) is the substrate or product of chemical A (or B). 9 for the relationship between chemical A and chemical B is not listed above."
    elif dataset_name == "cdr":
        task = "chemical disease relation extraction"
        task_info = "In each iteration, the user will provide a biomedical passage, followed by a question asking whether a chemical causes " \
                    "a disease. Then the user will provide a label indicating whether the chemical causes the disease based on the passage."
        class_info = "0 for the chemical does not cause the disease, 1 for the chemical causes the disease"
    elif dataset_name == "spouse":
        task = "spouse relation extraction"
        task_info = "In each iteration, the user will provide a passage, followed by a question asking whether two people are spouses." \
                    "Then the user will provide a label indicating whether the two people are spouses based on the passage."
        class_info = "0 for the two people are not spouses, 1 for the two people are spouses."
    elif dataset_name == "semeval":
        task = "sematic relationship classification"
        task_info = "In each iteration, the user will provide a passage, followed by a question asking the semantic relationship between" \
                    " two nominals. Then the user will provide a label indicating the relationship between the two nominals."
        class_info = "0 for an event or object A leads to an effect B. " \
                     "1 for an object A is a component of a larger whole B. " \
                     "2 for an object A is physically stored in a delineated area of space B. " \
                     "3 for an entity A is moving towards a destination B. " \
                     "4 for an entity A is coming or is derived from an origin B. " \
                     "5 for an agent B uses an instrument A. " \
                     "6 for a member A forms a nonfunctional part of a collection B. " \
                     "7 for a message A, written or spoken, is about a topic B. " \
                     "8 for a producer B causes a product A to exist. "
    elif dataset_name == "agnews":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a piece of news and a label indicating the topic of the news."
        class_info = "0 for world news, 1 for sports news, 2 for business news, 3 for science or high technology news."
    elif dataset_name == "trec":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a question and a label indicating the topic of the question."
        class_info = "0 for questions asking for description and abstract concept, 1 for questions asking for an entity (animal, plant, color, etc.), " \
                     "2 for questions asking about a person or a group of persons, 3 for questions asking for an abbreviation," \
                     "4 for questions asking for a location, 5 for questions asking for a number (data, postcode, etc.)."
    elif dataset_name == "medical_abstract":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a medical abstract and a label indicating the topic of the abstract based on the disease it focuses on."
        class_info = "0 for neoplasms diseases, 1 for digestive system diseases, 2 for nervous system diseases, 3 for cardiovascular diseases, 4 for general pathological conditions."
    elif dataset_name == "arxiv_abstract":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a paper abstract and a label indicating the topic of the abstract. "
        class_info = "0 for Computer Vision and Pattern Recognition, covering image processing, computer vision, pattern recognition, and scene understanding." \
                     "1 for Machine Learning, covering Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems," \
                     "and so on) including also robustness, explanation, fairness, and methodology. Also for machine learning paper with a statistical or theoretical grounding."
    elif dataset_name == 'sato':
        task='LF creation for Sato dataset'
        task_info='In each iteration, the user will provide a sequence and a label indicating which class the sequence belongs to'
        class_info="0 for name." \
        "1 for description." \
        "2 for team." \
        "3 for age." \
        "4 for type." \
        "5 for location." \
        "6 for year." \
        "7 for city." \
        "8 for rank." \
        "9 for status." \
        "10 for state." \
        "11 for category." \
        "12 for weight." \
        "13 for code." \
        "14 for club." \
        "15 for artist." \
        "16 for result." \
        "17 for position." \
        "18 for country." \
        "19 for album." \
        "20 for company." \
        "21 for class." \
        "22 for symbol." \
        "23 for notes." \
        "24 for address." \
        "25 for duration." \
        "26 for format." \
        "27 for county." \
        "28 for day." \
        "29 for gender." \
        "30 for industry." \
        "31 for sex." \
        "32 for product." \
        "33 for jockey." \
        "34 for region." \
        "35 for language." \
        "36 for area." \
        "37 for service." \
        "38 for teamName." \
        "39 for isbn." \
        "40 for fileSize." \
        "41 for grades." \
        "42 for publisher." \
        "43 for plays." \
        "44 for order." \
        "45 for origin." \
        "46 for elevation." \
        "47 for affiliation." \
        "48 for owner." \
        "49 for component." \
        "50 for genre." \
        "51 for manufacturer." \
        "52 for brand." \
        "53 for credit." \
        "54 for family." \
        "55 for depth." \
        "56 for classification." \
        "57 for collection." \
        "58 for command." \
        "59 for species." \
        "60 for nationality." \
        "61 for currency." \
        "62 for range." \
        "63 for birthDate." \
        "64 for ranking." \
        "65 for capacity." \
        "66 for birthPlace." \
        "67 for creator." \
        "68 for operator." \
        "69 for religion." \
        "70 for education." \
        "71 for person." \
        "72 for requirement." \
        "73 for director." \
        "74 for sales." \
        "75 for affiliate." \
        "76 for continent." \
        "77 for organisation." 
    elif dataset_name == 'turl':
        task='LF creation for Sato dataset'
        task_info='In each iteration, the user will provide a sequence and a label indicating which class the sequence belongs to'
        class_info= class_info="0 for people.person." \
"1 for location.location." \
"2 for organization.organization." \
"3 for sports.sports_team." \
"4 for sports.pro_athlete." \
"5 for soccer.football_team." \
"6 for time.event." \
"7 for location.country." \
"8 for location.citytown." \
"9 for government.political_party." \
"10 for location.administrative_division." \
"11 for sports.sports_league_season." \
"12 for soccer.football_player." \
"13 for sports.sports_league." \
"14 for government.politician." \
"15 for film.film." \
"16 for business.business_operation." \
"17 for time.recurring_event." \
"18 for music.artist." \
"19 for film.actor." \
"20 for music.album." \
"21 for tv.tv_program." \
"22 for architecture.structure." \
"23 for baseball.baseball_player." \
"24 for sports.professional_sports_team." \
"25 for tennis.tennis_player." \
"26 for basketball.basketball_team." \
"27 for music.composition." \
"28 for government.u_s_congressperson." \
"29 for tv.tv_actor." \
"30 for education.educational_institution." \
"31 for sports.sports_position." \
"32 for architecture.venue." \
"33 for sports.sports_facility." \
"34 for sports.cyclist." \
"35 for sports.golfer." \
"36 for location.hud_county_place." \
"37 for film.director." \
"38 for government.general_election." \
"39 for education.university." \
"40 for basketball.basketball_player." \
"41 for ice_hockey.hockey_player." \
"42 for ice_hockey.hockey_team." \
"43 for soccer.football_league." \
"44 for biology.organism_classification." \
"45 for baseball.baseball_league." \
"46 for american_football.football_player." \
"47 for music.group_member." \
"48 for olympics.olympic_event_competition." \
"49 for cricket.cricket_player." \
"50 for american_football.football_team." \
"51 for sports.sport." \
"52 for aviation.airport." \
"53 for book.author." \
"54 for sports.school_sports_team." \
"55 for baseball.baseball_team." \
"56 for location.us_county." \
"57 for royalty.noble_person." \
"58 for tv.tv_network." \
"59 for education.athletics_brand." \
"60 for music.record_label." \
"61 for tennis.tennis_tournament_champion." \
"62 for award.award_category." \
"63 for award.competition." \
"64 for aviation.aircraft_owner." \
"65 for language.human_language." \
"66 for aviation.airline." \
"67 for location.capital_of_administrative_division." \
"68 for sports.sports_championship_event." \
"69 for film.writer." \
"70 for music.composer." \
"71 for music.performance_role." \
"72 for government.election." \
"73 for fictional_universe.fictional_character." \
"74 for tennis.tennis_tournament." \
"75 for metropolitan_transit.transit_stop." \
"76 for military.military_person." \
"77 for broadcast.broadcast." \
"78 for internet.website." \
"79 for sports.tournament_event_competition." \
"80 for sports.multi_event_tournament." \
"81 for cvg.computer_videogame." \
"82 for book.written_work." \
"83 for boats.ship." \
"84 for broadcast.artist." \
"85 for award.award_ceremony." \
"86 for tv.tv_program_creator." \
"87 for location.us_state." \
"88 for transportation.road." \
"89 for cricket.cricket_team." \
"90 for automotive.model." \
"91 for soccer.football_award." \
"92 for education.school." \
"93 for people.ethnicity." \
"94 for ice_hockey.hockey_position." \
"95 for tv.tv_personality." \
"96 for automotive.company." \
"97 for government.legislative_session." \
"98 for baseball.baseball_position." \
"99 for film.producer." \
"100 for cricket.cricket_bowler." \
"101 for aviation.aircraft_model." \
"102 for soccer.football_position." \
"103 for media_common.media_genre." \
"104 for astronomy.celestial_object." \
"105 for sports.sports_championship." \
"106 for music.musical_group." \
"107 for broadcast.radio_station." \
"108 for military.military_unit." \
"109 for tv.tv_character." \
"110 for sports.boxer." \
"111 for olympics.olympic_games." \
"112 for book.periodical." \
"113 for architecture.building." \
"114 for martial_arts.martial_artist." \
"115 for people.family_member." \
"116 for soccer.football_league_season." \
"117 for film.production_company." \
"118 for royalty.monarch." \
"119 for chemistry.chemical_compound." \
"120 for award.recurring_competition." \
"121 for royalty.kingdom." \
"122 for biology.organism." \
"123 for music.lyricist." \
"124 for book.book." \
"125 for film.film_distributor." \
"126 for cvg.cvg_platform." \
"127 for military.rank." \
"128 for location.uk_statistical_location." \
"129 for cvg.cvg_developer." \
"130 for tv.tv_producer." \
"131 for basketball.basketball_conference." \
"132 for medicine.anatomical_structure." \
"133 for astronomy.orbital_relationship." \
"134 for spaceflight.astronaut." \
"135 for tv.tv_series_season." \
"136 for education.field_of_study." \
"137 for film.music_contributor." \
"138 for music.producer." \
"139 for business.consumer_company." \
"140 for geography.mountain." \
"141 for astronomy.star_system_body." \
"142 for film.film_genre." \
"143 for protected_sites.listed_site." \
"144 for computer.software." \
"145 for astronomy.astronomical_discovery." \
"146 for geography.body_of_water." \
"147 for book.magazine." \
"148 for government.government_office_or_title." \
"149 for cvg.cvg_publisher." \
"150 for organization.membership_organization." \
"151 for location.jp_prefecture." \
"152 for military.military_conflict." \
"153 for tv.tv_series_episode." \
"154 for metropolitan_transit.transit_line." \
"155 for basketball.basketball_coach." \
"156 for soccer.football_world_cup." \
"157 for astronomy.asteroid." \
"158 for government.us_president." \
"159 for award.award_presenting_organization." \
"160 for award.award." \
"161 for sports.sports_award_winner." \
"162 for soccer.fifa." \
"163 for award.hall_of_fame_inductee." \
"164 for boats.ship_class." \
"165 for comic_books.comic_book_character." \
"166 for basketball.basketball_position." \
"167 for film.film_character." \
"168 for tv.tv_director." \
"169 for tv.tv_writer." \
"170 for finance.currency." \
"171 for medicine.disease." \
"172 for rail.locomotive_class." \
"173 for theater.play." \
"174 for law.invention." \
"175 for government.governmental_body." \
"176 for geography.river." \
"177 for music.writer." \
"178 for american_football.football_coach." \
"179 for religion.religion." \
"180 for music.media_format." \
"181 for royalty.chivalric_order_member." \
"182 for location.province." \
"183 for broadcast.tv_station." \
"184 for food.food." \
"185 for meteorology.tropical_cyclone." \
"186 for cvg.cvg_genre." \
"187 for business.industry." \
"188 for military.armed_force." \
"189 for business.job_title." \
"190 for tv.tv_genre." \
"191 for meteorology.tropical_cyclone_season." \
"192 for geography.island." \
"193 for internet.website_owner." \
"194 for fictional_universe.fictional_organization." \
"195 for law.court." \
"196 for location.australian_local_government_area." \
"197 for business.product_category." \
"198 for music.genre." \
"199 for sports.sports_league_draft." \
"200 for computer.operating_system." \
"201 for theater.theater_actor." \
"202 for business.defunct_company." \
"203 for computer.software_license." \
"204 for location.in_state." \
"205 for book.periodical_subject." \
"206 for cricket.cricket_stadium." \
"207 for american_football.football_conference." \
"208 for music.musical_scale." \
"209 for medicine.drug_ingredient." \
"210 for soccer.football_team_manager." \
"211 for computer.computer." \
"212 for chemistry.chemical_element." \
"213 for amusement_parks.ride." \
"214 for award.award_discipline." \
"215 for celebrities.celebrity." \
"216 for royalty.noble_title." \
"217 for business.brand." \
"218 for medicine.drug." \
"219 for broadcast.genre." \
"220 for interests.collection_category." \
"221 for business.customer." \
"222 for government.election_campaign." \
"223 for organization.non_profit_organization." \
"224 for boats.ship_type." \
"225 for location.in_district." \
"226 for travel.accommodation." \
"227 for medicine.medical_treatment." \
"228 for metropolitan_transit.transit_system." \
"229 for location.australian_state." \
"230 for law.legal_case." \
"231 for location.uk_constituent_country." \
"232 for business.consumer_product." \
"233 for broadcast.tv_channel." \
"234 for broadcast.radio_format." \
"235 for location.region." \
"236 for religion.religious_leader." \
"237 for amusement_parks.park." \
"238 for exhibitions.exhibition_sponsor." \
"239 for sports.sports_award_type." \
"240 for military.military_post." \
"241 for education.fraternity_sorority." \
"242 for book.periodical_publisher." \
"243 for government.government_agency." \
"244 for medicine.muscle." \
"245 for biology.animal." \
"246 for music.music_video_director." \
"247 for visual_art.visual_artist." \
"248 for film.film_festival_focus." \
"249 for book.newspaper." \
"250 for architecture.architectural_structure_owner." \
"251 for music.instrument." \
"252 for astronomy.constellation." \
"253 for chess.chess_player." \
"254 for education.educational_degree." \

    if "lf_type" in kwargs:
        lf_type = kwargs["lf_type"]
    else:
        lf_type = "keyword"

    if "explanation" in kwargs:
        explanation = kwargs["explanation"]
    else:
        explanation = False

    if lf_type == "keyword":
        if explanation:
            interaction_format = """
                    After the user provides input, provide a step-by-step explanation to justify the user's provided label. 
                    Then identify a list of keywords that helps making prediction. The interaction format is as follows. 
                    Replace the text in brackets when you respond to user query.
                    User: 
                    [Input text]
                    LABEL: [Predicted label]
                    Response:
                    EXPLANATION: <Explain the reason process step by step>
                    KEYWORDS: <List of keywords>
                    """
        else:
            interaction_format = """
                                After the user provides input, identify a list of keywords that helps making prediction. 
                                The interaction format is as follows. Replace the text in brackets when you respond to user query.
                                User: 
                                [Input text]
                                LABEL: [Predicted label]
                                Response:
                                KEYWORDS: <List of keywords>
                                """
    else:
        if dataset_name == "cdr":
            regex_instruction = "In the regular expression, use {{A}} to represent the chemical and {{B}} to represent the disease that" \
                                " occur in the user's query. Use [SEP] to seperate multiple regular expressions."
        elif dataset_name == "chemprot":
            regex_instruction = "In the regular expression, use {{A}} to represent the first chemical and {{B}} to represent the second " \
                                "chemical that occur in the user's query. Use [SEP] to seperate multiple regular expressions."
        else:
            regex_instruction = ""

        if explanation:
            interaction_format = """
                                After the user provides input, provide a step-by-step explanation to justify the user's provided label. 
                                Then provide a regular expression such that if a passage matches the regex, it is likely to 
                                have the same label with the current input. {} If no regular expression can be identified, return NONE
                                for regular expression. The interaction format is as follows. Replace the text in brackets when you respond to user query.
                                User: 
                                [Input text]
                                LABEL: [Predicted label]
                                Response:
                                EXPLANATION: <Explain the reason process step by step>
                                REGEX: <List of regular expressions>
                                """.format(regex_instruction)
        else:
            interaction_format = """
                                After the user provides input, provide a regular expression such that if a passage matches the regex, it is likely to 
                                have the same label with the current input. {} If no regular expression can be identified, return NONE
                                for regular expression. The interaction format is as follows. Replace the text in brackets when you respond to user query.
                                User: 
                                [Input text]
                                LABEL: [Predicted label]
                                Response:
                                EXPLANATION: <Explain the reason process step by step>
                                REGEX: <List of regular expressions>
                                """.format(regex_instruction)
                            

    np.random.seed(kwargs['seed'])
    random.seed(kwargs['seed'])
    example_string = ""
    if example_per_class > 0:
        # use fixed examples
        # with open("/Users/chenjieli/Desktop/LLMDP/examples.json") as json_file:
        with open("/nfs/users/chenjie/LLMDP/examples.json") as json_file:
            example_dict = json.load(json_file)

        examples = example_dict[dataset_name]
        example_labels = []
        example_indices = []  # example indices in example file. NOT the original indices in validation set.
        for e in examples:
            example_labels.append(e["label"])
        
        for c in range(dataset.n_class):
            active_indices = np.nonzero(np.array(example_labels) == c)[0]
            assert len(active_indices) >= example_per_class
            selected_indices = np.random.choice(active_indices, example_per_class)
            example_indices += selected_indices.tolist()

        for idx in example_indices:
            user_input = examples[idx]["data"]
            label = examples[idx]["label"]

            if lf_type == "keyword":
                keywords = examples[idx]["keywords"]
                if explanation:
                    explanation = examples[idx]["explanation"]
                    example = "User:{}\nLABEL: {}\nResponse:\nExplanation: {}\nKEYWORDS: {}\n".format(
                        user_input, label, explanation, keywords)
                else:
                    example = "User:{}\nLABEL: {}\nResponse:\nKEYWORDS: {}\n".format(user_input, label, keywords)
            elif lf_type == "regex":
                regex = examples[idx]["regex"]
                if explanation:
                    explanation = examples[idx]["explanation"]
                    example = "User:{}\nResponse:\nExplanation: {}\nREGEX: {}\nLABEL: {}\n".format(
                        user_input, explanation, regex, label)
                else:
                    example = "User:{}\nLABEL: {}\nResponse:\nREGEX: {}\n".format(user_input, label, regex)

            example_string += example

    task_prompt = """
    TASK DESCRIPTION: 
    You are a helpful assistant who helps users in a {} task. {} ({})
    INTERACTION FORMAT: {}""".format(task, task_info, class_info, interaction_format)

    logger.warning("task_prompt:")
    logger.warning(task_prompt)
    logger.warning('example_string:')
    logger.warning(example_string)

    return task_prompt, example_string


def create_prompt(dataset_name, dataset, example_per_class=1, example_selection="random", **kwargs):
    """
    Create prompt for label function generation
    """

    logger.warning("creating prompt")
    if dataset_name == "youtube":
        task = "spam classification"
        task_info = "In each iteration, the user will provide a comment for a video. Please decide whether the comment is a spam."
        class_info = "0 for non-spam, 1 for spam"
    elif dataset_name == "sms":
        task = "spam classification"
        task_info = "In each iteration, the user will provide a text message. Please decide whether the message is a spam. Hint: promotional " \
                    "messages should also be considered as spam messages."
        class_info = "0 for non-spam, 1 for spam"
    elif dataset_name == "imdb":
        task = "sentiment analysis"
        task_info = "In each iteration, the user will provide a movie review. Please decide whether the review is positive or negative."
        class_info = "0 for negative, 1 for positive"
    elif dataset_name == "yelp":
        task = "sentiment analysis"
        task_info = "In each iteration, the user will provide a restaurant review. Please decide whether the review is positive or negative."
        class_info = "0 for negative, 1 for positive"
    elif dataset_name == "chemprot":
        task = "biomedical relation extraction"
        task_info = "In each iteration, the user will provide a biomedical statement, followed by a question asking the relationship between two chemicals occured in that statement." \
                    "Please decide the relationship between the two chemicals based on the statement."
        class_info = "0 for chemical B (or A) is part of chemical A (or B), 1 for chemical B (or A) is the regulator of chemical A (or B). " \
                     "2 for chemical B (or A) is the upregulator of chemical A (or B). 3 for chemical B (or A) is the downregulator of chemical A (or B)." \
                     "4 for chemical B (or A) is the agnoist of chemical A (or B). 5 for chemical B (or A) is the antagonist of chemical A (or B)." \
                     "6 for chemical B (or A) is the modulator of chemical A (or B). 7 for chemical B (or A) is the cofactor of chemical A (or B)." \
                     "8 for chemical B (or A) is the substrate or product of chemical A (or B). 9 for the relationship between chemical A and chemical B is not listed above."
    elif dataset_name == "cdr":
        task = "chemical disease relation extraction"
        task_info = "In each iteration, the user will provide a biomedical passage, followed by a question asking whether a chemical causes " \
                    "a disease. Please decide whether the chemical causes the disease based on the passage. Hint: please be rigorous when making" \
                    "the causal claim, that is, only return 1 if the passage explictly states that the chemical causes the disease, and return 0" \
                    "when it only indicate a possibility of causal relationship."
        class_info = "0 for the chemical does not cause the disease, 1 for the chemical causes the disease"
    elif dataset_name == "agnews":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a piece of news. Please classify the topic of the news into following categories."
        class_info = "0 for world news, 1 for sports news, 2 for business news, 3 for science or high technology news."
    elif dataset_name == "trec":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a question. Please classify the topic of the question into following categories."
        class_info = "0 for questions asking for description and abstract concept, 1 for questions asking for an entity (animal, plant, color, etc.), " \
                     "2 for questions asking about a person or a group of persons, 3 for questions asking for an abbreviation," \
                     "4 for questions asking for a location, 5 for questions asking for a number (data, postcode, etc.)."
    elif dataset_name == "medical_abstract":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a medical abstract. Please classify the topic of the abstract based on the disease it focuses on."
        class_info = "0 for neoplasms diseases, 1 for digestive system diseases, 2 for nervous system diseases, 3 for cardiovascular diseases, 4 for general pathological conditions."
    elif dataset_name == "spouse":
        task = "spouse relation extraction"
        task_info = "In each iteration, the user will provide a passage, followed by a question asking whether two people are spouses." \
                    "Please decide whether the two people are spouses based on the given passage."
        class_info = "0 for the two people are not spouses, 1 for the two people are spouses."
    elif dataset_name == "semeval":
        task = "sematic relationship classification"
        task_info = "In each iteration, the user will provide a passage, followed by a question asking the semantic relationship between" \
                    " two nominals. Please classify the semantic relationship between two nominals into one of the following categories based on the passage."
        class_info = "0 for an event or object A leads to an effect B. " \
                     "1 for an object A is a component of a larger whole B. " \
                     "2 for an object A is physically stored in a delineated area of space B. " \
                     "3 for an entity A is moving towards a destination B. " \
                     "4 for an entity A is coming or is derived from an origin B. " \
                     "5 for an agent B uses an instrument A. " \
                     "6 for a member A forms a nonfunctional part of a collection B. " \
                     "7 for a message A, written or spoken, is about a topic B. " \
                     "8 for a producer B causes a product A to exist. "
    elif dataset_name == "arxiv_abstract":
        task = "topic classification"
        task_info = "In each iteration, the user will provide a paper abstract. Please classify the topic of the abstract into following categories. "
        class_info = "0 for Computer Vision and Pattern Recognition, covering image processing, computer vision, pattern recognition, and scene understanding." \
                     "1 for Machine Learning, covering Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems," \
                     "and so on) including also robustness, explanation, fairness, and methodology. Also for machine learning paper with a statistical or theoretical grounding."
    elif dataset_name == 'sato':
        task='LF creation for Sato dataset'
        task_info='In each iteration, the user will provide a sequence and a label indicating which class the sequence belongs to'
        class_info="0 for name." \
    "1 for description." \
    "2 for team." \
    "3 for age." \
    "4 for type." \
    "5 for location." \
    "6 for year." \
    "7 for city." \
    "8 for rank." \
    "9 for status." \
    "10 for state." \
    "11 for category." \
    "12 for weight." \
    "13 for code." \
    "14 for club." \
    "15 for artist." \
    "16 for result." \
    "17 for position." \
    "18 for country." \
    "19 for album." \
    "20 for company." \
    "21 for class." \
    "22 for symbol." \
    "23 for notes." \
    "24 for address." \
    "25 for duration." \
    "26 for format." \
    "27 for county." \
    "28 for day." \
    "29 for gender." \
    "30 for industry." \
    "31 for sex." \
    "32 for product." \
    "33 for jockey." \
    "34 for region." \
    "35 for language." \
    "36 for area." \
    "37 for service." \
    "38 for teamName." \
    "39 for isbn." \
    "40 for fileSize." \
    "41 for grades." \
    "42 for publisher." \
    "43 for plays." \
    "44 for order." \
    "45 for origin." \
    "46 for elevation." \
    "47 for affiliation." \
    "48 for owner." \
    "49 for component." \
    "50 for genre." \
    "51 for manufacturer." \
    "52 for brand." \
    "53 for credit." \
    "54 for family." \
    "55 for depth." \
    "56 for classification." \
    "57 for collection." \
    "58 for command." \
    "59 for species." \
    "60 for nationality." \
    "61 for currency." \
    "62 for range." \
    "63 for birthDate." \
    "64 for ranking." \
    "65 for capacity." \
    "66 for birthPlace." \
    "67 for creator." \
    "68 for operator." \
    "69 for religion." \
    "70 for education." \
    "71 for person." \
    "72 for requirement." \
    "73 for director." \
    "74 for sales." \
    "75 for affiliate." \
    "76 for continent." \
    "77 for organisation."
    elif dataset_name == 'turl':
        task='LF creation for Sato dataset'
        task_info='In each iteration, the user will provide a sequence and a label indicating which class the sequence belongs to'
        class_info= class_info="0 for people.person." \
        "1 for location.location." \
        "2 for organization.organization." \
        "3 for sports.sports_team." \
        "4 for sports.pro_athlete." \
        "5 for soccer.football_team." \
        "6 for time.event." \
        "7 for location.country." \
        "8 for location.citytown." \
        "9 for government.political_party." \
        "10 for location.administrative_division." \
        "11 for sports.sports_league_season." \
        "12 for soccer.football_player." \
        "13 for sports.sports_league." \
        "14 for government.politician." \
        "15 for film.film." \
        "16 for business.business_operation." \
        "17 for time.recurring_event." \
        "18 for music.artist." \
        "19 for film.actor." \
        "20 for music.album." \
        "21 for tv.tv_program." \
        "22 for architecture.structure." \
        "23 for baseball.baseball_player." \
        "24 for sports.professional_sports_team." \
        "25 for tennis.tennis_player." \
        "26 for basketball.basketball_team." \
        "27 for music.composition." \
        "28 for government.u_s_congressperson." \
        "29 for tv.tv_actor." \
        "30 for education.educational_institution." \
        "31 for sports.sports_position." \
        "32 for architecture.venue." \
        "33 for sports.sports_facility." \
        "34 for sports.cyclist." \
        "35 for sports.golfer." \
        "36 for location.hud_county_place." \
        "37 for film.director." \
        "38 for government.general_election." \
        "39 for education.university." \
        "40 for basketball.basketball_player." \
        "41 for ice_hockey.hockey_player." \
        "42 for ice_hockey.hockey_team." \
        "43 for soccer.football_league." \
        "44 for biology.organism_classification." \
        "45 for baseball.baseball_league." \
        "46 for american_football.football_player." \
        "47 for music.group_member." \
        "48 for olympics.olympic_event_competition." \
        "49 for cricket.cricket_player." \
        "50 for american_football.football_team." \
        "51 for sports.sport." \
        "52 for aviation.airport." \
        "53 for book.author." \
        "54 for sports.school_sports_team." \
        "55 for baseball.baseball_team." \
        "56 for location.us_county." \
        "57 for royalty.noble_person." \
        "58 for tv.tv_network." \
        "59 for education.athletics_brand." \
        "60 for music.record_label." \
        "61 for tennis.tennis_tournament_champion." \
        "62 for award.award_category." \
        "63 for award.competition." \
        "64 for aviation.aircraft_owner." \
        "65 for language.human_language." \
        "66 for aviation.airline." \
        "67 for location.capital_of_administrative_division." \
        "68 for sports.sports_championship_event." \
        "69 for film.writer." \
        "70 for music.composer." \
        "71 for music.performance_role." \
        "72 for government.election." \
        "73 for fictional_universe.fictional_character." \
        "74 for tennis.tennis_tournament." \
        "75 for metropolitan_transit.transit_stop." \
        "76 for military.military_person." \
        "77 for broadcast.broadcast." \
        "78 for internet.website." \
        "79 for sports.tournament_event_competition." \
        "80 for sports.multi_event_tournament." \
        "81 for cvg.computer_videogame." \
        "82 for book.written_work." \
        "83 for boats.ship." \
        "84 for broadcast.artist." \
        "85 for award.award_ceremony." \
        "86 for tv.tv_program_creator." \
        "87 for location.us_state." \
        "88 for transportation.road." \
        "89 for cricket.cricket_team." \
        "90 for automotive.model." \
        "91 for soccer.football_award." \
        "92 for education.school." \
        "93 for people.ethnicity." \
        "94 for ice_hockey.hockey_position." \
        "95 for tv.tv_personality." \
        "96 for automotive.company." \
        "97 for government.legislative_session." \
        "98 for baseball.baseball_position." \
        "99 for film.producer." \
        "100 for cricket.cricket_bowler." \
        "101 for aviation.aircraft_model." \
        "102 for soccer.football_position." \
        "103 for media_common.media_genre." \
        "104 for astronomy.celestial_object." \
        "105 for sports.sports_championship." \
        "106 for music.musical_group." \
        "107 for broadcast.radio_station." \
        "108 for military.military_unit." \
        "109 for tv.tv_character." \
        "110 for sports.boxer." \
        "111 for olympics.olympic_games." \
        "112 for book.periodical." \
        "113 for architecture.building." \
        "114 for martial_arts.martial_artist." \
        "115 for people.family_member." \
        "116 for soccer.football_league_season." \
        "117 for film.production_company." \
        "118 for royalty.monarch." \
        "119 for chemistry.chemical_compound." \
        "120 for award.recurring_competition." \
        "121 for royalty.kingdom." \
        "122 for biology.organism." \
        "123 for music.lyricist." \
        "124 for book.book." \
        "125 for film.film_distributor." \
        "126 for cvg.cvg_platform." \
        "127 for military.rank." \
        "128 for location.uk_statistical_location." \
        "129 for cvg.cvg_developer." \
        "130 for tv.tv_producer." \
        "131 for basketball.basketball_conference." \
        "132 for medicine.anatomical_structure." \
        "133 for astronomy.orbital_relationship." \
        "134 for spaceflight.astronaut." \
        "135 for tv.tv_series_season." \
        "136 for education.field_of_study." \
        "137 for film.music_contributor." \
        "138 for music.producer." \
        "139 for business.consumer_company." \
        "140 for geography.mountain." \
        "141 for astronomy.star_system_body." \
        "142 for film.film_genre." \
        "143 for protected_sites.listed_site." \
        "144 for computer.software." \
        "145 for astronomy.astronomical_discovery." \
        "146 for geography.body_of_water." \
        "147 for book.magazine." \
        "148 for government.government_office_or_title." \
        "149 for cvg.cvg_publisher." \
        "150 for organization.membership_organization." \
        "151 for location.jp_prefecture." \
        "152 for military.military_conflict." \
        "153 for tv.tv_series_episode." \
        "154 for metropolitan_transit.transit_line." \
        "155 for basketball.basketball_coach." \
        "156 for soccer.football_world_cup." \
        "157 for astronomy.asteroid." \
        "158 for government.us_president." \
        "159 for award.award_presenting_organization." \
        "160 for award.award." \
        "161 for sports.sports_award_winner." \
        "162 for soccer.fifa." \
        "163 for award.hall_of_fame_inductee." \
        "164 for boats.ship_class." \
        "165 for comic_books.comic_book_character." \
        "166 for basketball.basketball_position." \
        "167 for film.film_character." \
        "168 for tv.tv_director." \
        "169 for tv.tv_writer." \
        "170 for finance.currency." \
        "171 for medicine.disease." \
        "172 for rail.locomotive_class." \
        "173 for theater.play." \
        "174 for law.invention." \
        "175 for government.governmental_body." \
        "176 for geography.river." \
        "177 for music.writer." \
        "178 for american_football.football_coach." \
        "179 for religion.religion." \
        "180 for music.media_format." \
        "181 for royalty.chivalric_order_member." \
        "182 for location.province." \
        "183 for broadcast.tv_station." \
        "184 for food.food." \
        "185 for meteorology.tropical_cyclone." \
        "186 for cvg.cvg_genre." \
        "187 for business.industry." \
        "188 for military.armed_force." \
        "189 for business.job_title." \
        "190 for tv.tv_genre." \
        "191 for meteorology.tropical_cyclone_season." \
        "192 for geography.island." \
        "193 for internet.website_owner." \
        "194 for fictional_universe.fictional_organization." \
        "195 for law.court." \
        "196 for location.australian_local_government_area." \
        "197 for business.product_category." \
        "198 for music.genre." \
        "199 for sports.sports_league_draft." \
        "200 for computer.operating_system." \
        "201 for theater.theater_actor." \
        "202 for business.defunct_company." \
        "203 for computer.software_license." \
        "204 for location.in_state." \
        "205 for book.periodical_subject." \
        "206 for cricket.cricket_stadium." \
        "207 for american_football.football_conference." \
        "208 for music.musical_scale." \
        "209 for medicine.drug_ingredient." \
        "210 for soccer.football_team_manager." \
        "211 for computer.computer." \
        "212 for chemistry.chemical_element." \
        "213 for amusement_parks.ride." \
        "214 for award.award_discipline." \
        "215 for celebrities.celebrity." \
        "216 for royalty.noble_title." \
        "217 for business.brand." \
        "218 for medicine.drug." \
        "219 for broadcast.genre." \
        "220 for interests.collection_category." \
        "221 for business.customer." \
        "222 for government.election_campaign." \
        "223 for organization.non_profit_organization." \
        "224 for boats.ship_type." \
        "225 for location.in_district." \
        "226 for travel.accommodation." \
        "227 for medicine.medical_treatment." \
        "228 for metropolitan_transit.transit_system." \
        "229 for location.australian_state." \
        "230 for law.legal_case." \
        "231 for location.uk_constituent_country." \
        "232 for business.consumer_product." \
        "233 for broadcast.tv_channel." \
        "234 for broadcast.radio_format." \
        "235 for location.region." \
        "236 for religion.religious_leader." \
        "237 for amusement_parks.park." \
        "238 for exhibitions.exhibition_sponsor." \
        "239 for sports.sports_award_type." \
        "240 for military.military_post." \
        "241 for education.fraternity_sorority." \
        "242 for book.periodical_publisher." \
        "243 for government.government_agency." \
        "244 for medicine.muscle." \
        "245 for biology.animal." \
        "246 for music.music_video_director." \
        "247 for visual_art.visual_artist." \
        "248 for film.film_festival_focus." \
        "249 for book.newspaper." \
        "250 for architecture.architectural_structure_owner." \
        "251 for music.instrument." \
        "252 for astronomy.constellation." \
        "253 for chess.chess_player." \
        "254 for education.educational_degree." 
    if "lf_type" in kwargs:
        lf_type = kwargs["lf_type"]
    else:
        lf_type = "keyword"

    if "explanation" in kwargs:
        explanation = kwargs["explanation"]
    else:
        explanation = False

    if "limited_sys_instance" in kwargs:
        limited_sys_instance=True 
    if "sys_limit_cnt" in kwargs:
        sys_limit_cnt=kwargs['sys_limit_cnt']
    
    if "user_provide_instance_label" in kwargs:
        user_provide_instance_label=kwargs['user_provide_instance_label']
    else:
        user_provide_instance_label=False

    if lf_type == "keyword":
        if explanation:
            if(not user_provide_instance_label):
                interaction_format = """
            After the user provides input, explain your reason process step by step. Then identify a list of keywords that helps
            making prediction. Finally, provide the class label for the input. The interaction format is as follows. Replace the 
            text in brackets when you respond to user query.
            User: 
            [Input text]
            Response:
            EXPLANATION: <Explain the reason process step by step>
            KEYWORDS: <List of keywords>
            LABEL: <Predicted label>
            """
            else:
                interaction_format = """
            After the user provides input, explain your reason process step by step. Then identify a list of keywords that helps
            making prediction. The interaction format is as follows. Replace the 
            text in brackets when you respond to user query.
            User: 
            [Input text]
            The label for this text:
            [LABEL]
            Response:
            EXPLANATION: <Explain the reason process step by step>
            KEYWORDS: <List of keywords>
            """
            
        else:
            if not user_provide_instance_label:
                interaction_format = """
                After the user provides input, identify a list of keywords that helps making prediction. Then provide the class label 
                for the input. The interaction format is as follows. Replace the text in brackets when you respond to user query.
                User: 
                [Input text]
                Response:
                KEYWORDS: <List of keywords>
                LABEL: <Predicted label>
                """
            else:
                interaction_format = """
                After the user provides input, identify a list of keywords that helps making prediction.
                The interaction format is as follows. Replace the text in brackets when you respond to user query.
                User: 
                [Input text]
                The label for this text:
                [LABEL]
                Response:
                KEYWORDS: <List of keywords>
                """


    elif lf_type == "regex":
        if dataset_name in ["cdr", "chemprot", "spouse", "semeval"]:
            regex_instruction = "In the regular expression, use {{A}} to represent the first entity and {{B}} to represent " \
                                "the second entity that occur in the user's query. Use [SEP] to seperate multiple regular expressions."
        else:
            regex_instruction = "Use [SEP] to seperate multiple regular expressions."

        if explanation:
            interaction_format = """
        After the user provides input, explain your reason process step by step. Then provide a regular expression such that
        if a passage matches the regex, it is likely to have the same label with the current input.{} If no regular expression
        can be identified, return NONE for regular expression. Finally, provide the class label for the input. The interaction 
        format is as follows. Replace the text in brackets when you respond to user query.
        User: 
        [Input text]
        Response:
        EXPLANATION: <Explain the reason process step by step>
        REGEX: <List of regular expressions>
        LABEL: <Predicted label>
        """.format(regex_instruction)
        else:
            interaction_format = """
        After the user provides input, provide a regular expression such that if a passage matches the regex, it is likely to 
        have the same label with the current input. {} If no regular expression can be identified, return NONE for regular expression. 
        Finally, provide the class label for the input. The interaction format is as follows. Replace the text in brackets when 
        you respond to user query.
        User: 
        [Input text]
        Response:
        REGEX: <List of regular expressions>
        LABEL: <Predicted label>
        """.format(regex_instruction)
    else:
        raise NotImplementedError(f"LF type {lf_type} not supported.")

    example_string = ""
    example_cnt = 0
    np.random.seed(kwargs['seed'])
    random.seed(kwargs['seed'])
    if example_per_class > 0:
        # use fixed examples
        # with open("/Users/chenjieli/Desktop/LLMDP/examples.json") as json_file:
        with open("/nfs/users/chenjie/LLMDP/examples.json") as json_file:
            example_dict = json.load(json_file)

        examples = example_dict[dataset_name]
        example_labels = []
        example_indices = []  # example indices in example file. NOT the original indices in validation set.
        for e in examples:
            example_labels.append(e["label"])

        # print(f"dataset.n_class: {dataset.n_class}")
        # print(f"example labels : {example_labels}")
        if(limited_sys_instance):
            if(sys_limit_cnt<=dataset.n_class):
                classses = random.sample(list(range(dataset.n_class)), sys_limit_cnt)
                for c in classses:
                    active_indices = np.nonzero(np.array(example_labels) == c)[0]

                    # print(f"c:{c}, len(active_indices): {len(active_indices)}")
                    selected_indices = np.random.choice(active_indices, 1)
                    example_indices += selected_indices.tolist()
            else:
                logger.warning("invalid sys limit cnt")
                exit()
        else:
            for c in range(dataset.n_class):
                active_indices = np.nonzero(np.array(example_labels) == c)[0]
                # print(f"c:{c}, len(active_indices): {len(active_indices)}")
                assert len(active_indices) >= example_per_class
                selected_indices = np.random.choice(active_indices, example_per_class)
                example_indices += selected_indices.tolist()

        for idx in example_indices:
            user_input = examples[idx]["data"]
            label = examples[idx]["label"]

            if lf_type == "keyword":
                keywords = examples[idx]["keywords"]
                if explanation:
                    explanation = examples[idx]["explanation"]
                    if(user_provide_instance_label):
                        example = "User:{}\nThe label for this text:\n{}\n Response:\nExplanation: {}\n KEWYWORDS: {}\n".format(
                            user_input, label, explanation, keywords
                        )
                    else:
                        example = "User:{}\nResponse:\nExplanation: {}\nKEYWORDS: {}\nLABEL: {}\n".format(
                        user_input, explanation, keywords, label)

                else:
                    if(user_provide_instance_label):
                        example = "User:{}\n The label for this text:\n{} \n Response:\nKEYWORDS: {}\n".format(user_input, label, keywords)
                    else:
                        example = "User:{}\n Response:\n {}\n KEWYWORDS: {}\n".format(
                            user_input, label, keywords
                        )

            elif lf_type == "regex":
                regex = examples[idx]["regex"]
                if explanation:
                    explanation = examples[idx]["explanation"]
                    example = "User:{}\nResponse:\nExplanation: {}\nREGEX: {}\nLABEL: {}\n".format(
                        user_input, explanation, regex, label)
                else:
                    example = "User:{}\nResponse:\nREGEX: {}\nLABEL: {}\n".format(user_input, regex, label)

            example_string += example
            example_cnt+=1



    task_prompt = """
TASK DESCRIPTION: 
You are a helpful assistant who helps users in a {} task. {} ({})
INTERACTION FORMAT: {}""".format(task, task_info, class_info, interaction_format)
    logger.warning("system prompt")
    logger.warning(task_prompt)
    logger.warning("example_string")
    logger.warning(example_string)
    # exit()

    return task_prompt, example_string




