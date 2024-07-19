import streamlit as st

import pickle

import pandas as pd
import numpy as np
import re

import math
from urllib.parse import urlparse

with open('dtc.pkl', 'rb') as file:
    dtc = pickle.load(file)

def extract_features(url, feature_columns):
    url_length = len(url)
    number_of_dots_in_url = url.count('.')
    having_repeated_digits_in_url = 1 if re.search(r'(\d)\1', url) else 0
    number_of_digits_in_url = sum(c.isdigit() for c in url)
    number_of_special_char_in_url = len(re.findall(r'[^a-zA-Z0-9]', url))
    number_of_hyphens_in_url = url.count('-')
    number_of_underline_in_url = url.count('_')
    number_of_slash_in_url = url.count('/')
    number_of_questionmark_in_url = url.count('?')
    number_of_equal_in_url = url.count('=')
    number_of_at_in_url = url.count('@')
    number_of_dollar_in_url = url.count('$')
    number_of_exclamation_in_url = url.count('!')
    number_of_hashtag_in_url = url.count('#')
    number_of_percent_in_url = url.count('%')

    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query
    fragment = parsed_url.fragment
    subdomain = domain.split('.')[0] if len(domain.split('.')) > 2 else ''

    domain_length = len(domain)
    number_of_dots_in_domain = domain.count('.')
    number_of_hyphens_in_domain = domain.count('-')
    having_special_characters_in_domain = 1 if re.search(r'[^a-zA-Z0-9]', domain) else 0
    number_of_special_characters_in_domain = len(re.findall(r'[^a-zA-Z0-9]', domain))
    having_digits_in_domain = 1 if any(c.isdigit() for c in domain) else 0
    number_of_digits_in_domain = sum(c.isdigit() for c in domain)
    having_repeated_digits_in_domain = 1 if re.search(r'(\d)\1', domain) else 0

    number_of_subdomains = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
    having_dot_in_subdomain = 1 if '.' in subdomain else 0
    having_hyphen_in_subdomain = 1 if '-' in subdomain else 0
    average_subdomain_length = len(subdomain) / number_of_subdomains if number_of_subdomains > 0 else 0
    average_number_of_dots_in_subdomain = subdomain.count('.') / number_of_subdomains if number_of_subdomains > 0 else 0
    average_number_of_hyphens_in_subdomain = subdomain.count(
        '-') / number_of_subdomains if number_of_subdomains > 0 else 0
    having_special_characters_in_subdomain = 1 if re.search(r'[^a-zA-Z0-9]', subdomain) else 0
    number_of_special_characters_in_subdomain = len(re.findall(r'[^a-zA-Z0-9]', subdomain))
    having_digits_in_subdomain = 1 if any(c.isdigit() for c in subdomain) else 0
    number_of_digits_in_subdomain = sum(c.isdigit() for c in subdomain)
    having_repeated_digits_in_subdomain = 1 if re.search(r'(\d)\1', subdomain) else 0

    having_path = 1 if path else 0
    path_length = len(path)
    having_query = 1 if query else 0
    having_fragment = 1 if fragment else 0
    having_anchor = 1 if '#' in url else 0

    entropy_of_url = -sum(p * math.log(p, 2) for p in (float(url.count(c)) / len(url) for c in set(url)))
    entropy_of_domain = -sum(p * math.log(p, 2) for p in (float(domain.count(c)) / len(domain) for c in set(domain)))

    features = {
        'url_length': url_length,
        'number_of_dots_in_url': number_of_dots_in_url,
        'having_repeated_digits_in_url': having_repeated_digits_in_url,
        'number_of_digits_in_url': number_of_digits_in_url,
        'number_of_special_char_in_url': number_of_special_char_in_url,
        'number_of_hyphens_in_url': number_of_hyphens_in_url,
        'number_of_underline_in_url': number_of_underline_in_url,
        'number_of_slash_in_url': number_of_slash_in_url,
        'number_of_questionmark_in_url': number_of_questionmark_in_url,
        'number_of_equal_in_url': number_of_equal_in_url,
        'number_of_at_in_url': number_of_at_in_url,
        'number_of_dollar_in_url': number_of_dollar_in_url,
        'number_of_exclamation_in_url': number_of_exclamation_in_url,
        'number_of_hashtag_in_url': number_of_hashtag_in_url,
        'number_of_percent_in_url': number_of_percent_in_url,
        'domain_length': domain_length,
        'number_of_dots_in_domain': number_of_dots_in_domain,
        'number_of_hyphens_in_domain': number_of_hyphens_in_domain,
        'having_special_characters_in_domain': having_special_characters_in_domain,
        'number_of_special_characters_in_domain': number_of_special_characters_in_domain,
        'having_digits_in_domain': having_digits_in_domain,
        'number_of_digits_in_domain': number_of_digits_in_domain,
        'having_repeated_digits_in_domain': having_repeated_digits_in_domain,
        'number_of_subdomains': number_of_subdomains,
        'having_dot_in_subdomain': having_dot_in_subdomain,
        'having_hyphen_in_subdomain': having_hyphen_in_subdomain,
        'average_subdomain_length': average_subdomain_length,
        'average_number_of_dots_in_subdomain': average_number_of_dots_in_subdomain,
        'average_number_of_hyphens_in_subdomain': average_number_of_hyphens_in_subdomain,
        'having_special_characters_in_subdomain': having_special_characters_in_subdomain,
        'number_of_special_characters_in_subdomain': number_of_special_characters_in_subdomain,
        'having_digits_in_subdomain': having_digits_in_subdomain,
        'number_of_digits_in_subdomain': number_of_digits_in_subdomain,
        'having_repeated_digits_in_subdomain': having_repeated_digits_in_subdomain,
        'having_path': having_path,
        'path_length': path_length,
        'having_query': having_query,
        'having_fragment': having_fragment,
        'having_anchor': having_anchor,
        'entropy_of_url': entropy_of_url,
        'entropy_of_domain': entropy_of_domain
    }

    # Convert the features dictionary to a DataFrame with the same columns as the training data
    features_df = pd.DataFrame([features], columns=feature_columns)
    return features_df
feature_columns = ['url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url', 'number_of_digits_in_url',
                   'number_of_special_char_in_url', 'number_of_hyphens_in_url', 'number_of_underline_in_url',
                   'number_of_slash_in_url', 'number_of_questionmark_in_url', 'number_of_equal_in_url', 'number_of_at_in_url',
                   'number_of_dollar_in_url', 'number_of_exclamation_in_url', 'number_of_hashtag_in_url', 'number_of_percent_in_url',
                   'domain_length', 'number_of_dots_in_domain', 'number_of_hyphens_in_domain', 'having_special_characters_in_domain',
                   'number_of_special_characters_in_domain', 'having_digits_in_domain', 'number_of_digits_in_domain',
                   'having_repeated_digits_in_domain', 'number_of_subdomains', 'having_dot_in_subdomain', 'having_hyphen_in_subdomain',
                   'average_subdomain_length', 'average_number_of_dots_in_subdomain', 'average_number_of_hyphens_in_subdomain',
                   'having_special_characters_in_subdomain', 'number_of_special_characters_in_subdomain', 'having_digits_in_subdomain',
                   'number_of_digits_in_subdomain', 'having_repeated_digits_in_subdomain', 'having_path', 'path_length',
                   'having_query', 'having_fragment', 'having_anchor', 'entropy_of_url', 'entropy_of_domain']


st.title("Phishing URL Detector")

input_url=st.text_input("Enter the URL")
if st.button('Predict'):
    # url extraction
    extracted = extract_features(input_url, feature_columns)

    # predict
    result = dtc.predict(extracted)
    # display
    if result == 1:
        st.header("Beware , this URL might not be safe")
    else:
        st.header("URL is safe to work with")



