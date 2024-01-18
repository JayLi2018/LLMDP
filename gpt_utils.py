import numpy as np
import json
import re
import logging
import LLMDP_chenjie.logconfig
import logging

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


def create_user_prompt(example_prompt, dataset_name, dataset, query_idx):
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

    else:
        text = dataset.examples[query_idx]["text"]
        user_prompt = "{} User: {}\n Response: ".format(example_prompt, text)
    return user_prompt


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
        "2 for age." \
        "3 for city." \
        "4 for state." \
        "5 for status." \
        "6 for rank." \
        "7 for year." \
        "8 for team." \
        "9 for weight." \
        "10 for location." \
        "11 for type." \
        "12 for code." \
        "13 for club." \
        "14 for category." \
        "15 for artist." \
        "16 for album." \
        "17 for position." \
        "18 for result." \
        "19 for company." \
        "20 for country." \
        "21 for class." \
        "22 for symbol." \
        "23 for notes." \
        "24 for address." \
        "25 for gender." \
        "26 for county." \
        "27 for format." \
        "28 for sex." \
        "29 for duration."

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

    example_string = ""
    if example_per_class > 0:
        # use fixed examples
        with open("/Users/chenjieli/Desktop/LLMDP_chenjie/examples.json") as json_file:
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
            "2 for age." \
            "3 for city." \
            "4 for state." \
            "5 for status." \
            "6 for rank." \
            "7 for year." \
            "8 for team." \
            "9 for weight." \
            "10 for location." \
            "11 for type." \
            "12 for code." \
            "13 for club." \
            "14 for category." \
            "15 for artist." \
            "16 for album." \
            "17 for position." \
            "18 for result." \
            "19 for company." \
            "20 for country." \
            "21 for class." \
            "22 for symbol." \
            "23 for notes." \
            "24 for address." \
            "25 for gender." \
            "26 for county." \
            "27 for format." \
            "28 for sex." \
            "29 for duration."

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
        After the user provides input, identify a list of keywords that helps making prediction. Then provide the class label 
        for the input. The interaction format is as follows. Replace the text in brackets when you respond to user query.
        User: 
        [Input text]
        Response:
        KEYWORDS: <List of keywords>
        LABEL: <Predicted label>
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
    if example_per_class > 0:
        # use fixed examples
        with open("/Users/chenjieli/Desktop/LLMDP_chenjie/examples.json") as json_file:
            example_dict = json.load(json_file)

        examples = example_dict[dataset_name]
        example_labels = []
        example_indices = []  # example indices in example file. NOT the original indices in validation set.
        for e in examples:
            example_labels.append(e["label"])

        # print(f"dataset.n_class: {dataset.n_class}")
        # print(f"example labels : {example_labels}")

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
                    example = "User:{}\nResponse:\nExplanation: {}\nKEYWORDS: {}\nLABEL: {}\n".format(
                        user_input, explanation, keywords, label)
                else:
                    example = "User:{}\nResponse:\nKEYWORDS: {}\nLABEL: {}\n".format(user_input, keywords, label)
            elif lf_type == "regex":
                regex = examples[idx]["regex"]
                if explanation:
                    explanation = examples[idx]["explanation"]
                    example = "User:{}\nResponse:\nExplanation: {}\nREGEX: {}\nLABEL: {}\n".format(
                        user_input, explanation, regex, label)
                else:
                    example = "User:{}\nResponse:\nREGEX: {}\nLABEL: {}\n".format(user_input, regex, label)

            example_string += example

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




