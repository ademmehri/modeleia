from flask import Flask, request, jsonify
import pdfplumber
import spacy
from spacy.matcher import PhraseMatcher
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import spacy
from spellchecker import SpellChecker
app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app ,resources={r"/*": {"origins": "https://beecoders-job.vercel.app"}})
# Charger le modèle spaCy multilingue
nlp_multilingue = spacy.load("xx_ent_wiki_sm")

# Fonction pour extraire le texte d'un PDF avec pdfplumber
def extraire_texte_pdf_avec_pdfplumber(chemin_du_pdf):
    with pdfplumber.open(chemin_du_pdf) as pdf:
        texte_complet = ""
        for page in pdf.pages:
            texte_complet += page.extract_text()
    return texte_complet

# Fonction pour vérifier si un texte contient des mots-clés
def contient_mots_cles(texte, mots_cles):
    # Créer le PhraseMatcher avec les mots-clés
    matcher = PhraseMatcher(nlp_multilingue.vocab)
    patterns = [nlp_multilingue(mot.lower()) for mot in mots_cles]
    matcher.add("MotsCles", None, *patterns)
    
    # Traiter le texte avec le modèle spaCy multilingue
    doc = nlp_multilingue(texte)

    # Utiliser le Matcher pour trouver les occurrences des mots-clés
    matches = matcher(doc)

    # Vérifier si tous les mots-clés sont présents dans le texte
    mots_cles_present = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        mots_cles_present.add(span.text.lower())

    return set(mots_cles) == mots_cles_present

#Recomandation USER
def calculer_similarite_competences(offre_competences, candidat_competences):
    # Utiliser les compétences de l'offre comme document de référence pour le TF-IDF
    document_reference = ' '.join(offre_competences)
    
    # Initialiser le transformateur TF-IDF avec le document de référence
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit([document_reference])
    
    # Calculer les vecteurs TF-IDF pour les compétences de l'offre et du candidat
    offre_tfidf = tfidf_vectorizer.transform([' '.join(offre_competences)])
    candidat_tfidf = tfidf_vectorizer.transform([' '.join(candidat_competences)])
    
    # Calculer la similarité cosinus entre les vecteurs TF-IDF
    similarite = cosine_similarity(offre_tfidf, candidat_tfidf)[0][0]
    if similarite>1.0:
        similarite+=len(candidat_competences)-len(offre_competences)
    return similarite
#Recomandation RH
def calculer_similarite_competences_et_experience(offre_competences, candidat_competences, offre_experience, candidat_experience):
    print(offre_competences)
    print('this')
    print( candidat_competences)
    document_reference = ' '.join(offre_competences)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit([document_reference])
    
    offre_tfidf = tfidf_vectorizer.transform([' '.join(offre_competences)])
    candidat_tfidf = tfidf_vectorizer.transform([' '.join(candidat_competences)])
    
    similarite_competences = cosine_similarity(offre_tfidf, candidat_tfidf)[0][0]
    
    # Convertir l'expérience en un nombre
    try:
        offre_exp = float(offre_experience.split()[0])
        candidat_exp = float(candidat_experience.split()[0])
        print("zzz")
        print(offre_exp)
        print(candidat_exp)
    except ValueError:  
        return 0  # Retourner 0 si l'expérience n'est pas un nombre
    
    # Calculer la différence d'expérience
    diff_experience = candidat_exp - offre_exp
    print( diff_experience )
    # Seuil pour la similarité des compétences
    seuil_similarite = 0.2  # Ajuster ce seuil selon vos besoins
    
    # Pondérations
    poids_experience = 0.05
    poids_competences = 1
    
    # Combinaison linéaire de la similarité des compétences et de l'expérience
    similarite_totale = (similarite_competences * poids_competences) + (diff_experience * poids_experience)
    print(similarite_totale)
    # Si la similarité des compétences est inférieure au seuil, attribuer un score très faible
    if similarite_competences < seuil_similarite:
        similarite_totale = -9999  # Score très faible
    
    return similarite_totale
def recommander_meilleurs_candidats_RH(offre, candidats, nombre_recommandations=3):
    similarites = []
   
    for candidat in candidats:
        print("ok")
        simalirité = calculer_similarite_competences_et_experience(
            offre['competence'], 
            candidat['competences'],
            offre['experience'], 
            candidat['experience']
        )
        print('rr')
        similarites.append((candidat, simalirité))
    
    similarites = sorted(similarites, key=lambda x: x[1], reverse=True)
    
    # Filtrer les candidats ayant un score très faible
    filtered_similarites = [(c, s) for c, s in similarites if s > -9999]
    
    meilleurs_candidats = [candidat[0] for candidat in filtered_similarites[:nombre_recommandations]]
    
    return meilleurs_candidats
@app.route('/meilleurecandidats', methods=['POST'])
def recommander_meilleurs_candidatasoffre():
    try:
        data = request.json
        offre = data['offre']
        candidats = data['candidats']
        nb=data['nbcandidat']
        
        # Retourner les meilleurs candidats
        meilleurs_candidats =recommander_meilleurs_candidats_RH(offre,candidats,nb)
        print(meilleurs_candidats )
        print(jsonify({"meilleurs_candidats": meilleurs_candidats}))
        return jsonify(meilleurs_candidats), 200
    except Exception as e:
        logging.error(f"Erreur dans la fonction meillei : {e}")
        return jsonify({"error": "Une erreur est survenue"}), 500
@app.route('/analyser-cv', methods=['POST'])
def analyser_cv():
    try:
        # Récupérer les données de la requête
        mots_cles_str = request.form.get('mots_cles')
        print(mots_cles_str)
        mots_cles = mots_cles_str.split(',') if mots_cles_str else []
        # Convertir chaque mot-clé en minuscules
        mots_cles_en_minuscules = [mot.strip().lower() for mot in mots_cles]
        fichier_pdf = request.files['fichier_pdf']
        # Extraire le texte du CV
        texte_du_cv_pdf = extraire_texte_pdf_avec_pdfplumber(fichier_pdf)
        # Vérifier si le texte contient des mots-clés
        resultat_verification = contient_mots_cles(texte_du_cv_pdf.lower(),mots_cles_en_minuscules)
        print(resultat_verification)
        # Retourner le résultat
        return jsonify(resultat_verification)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/filtrage-offre', methods=['POST'])
def recommander_meilleurs_offre():
    try:
        data = request.json
        offres = data['offres']
        candidat = data['candidat']
        nb=data['nbcandidat']
        print("ok")
        # Calculer la similarité des compétences et de l'expérience pour chaque candidat
        similarites = []
        for offre in offres:
            similarite_competences = calculer_similarite_competences(offre['competence'], candidat['competences'])
            similarite_totale = similarite_competences
            print(candidat['competences'])
            print(similarite_totale)
            if similarite_totale>0:
                similarites.append((offre, similarite_totale))
        # Classer les candidats par similarité
        similarites = sorted(similarites, key=lambda x: x[1], reverse=True)
    
        # Retourner les meilleurs candidats
        meilleurs_offres = [offre[0] for offre in similarites[:nb]]
       
        
        return jsonify(meilleurs_offres), 200
    except Exception as e:
        logging.error(f"Erreur dans la fonction filtrage_offre : {e}")
        return jsonify({"error": "Une erreur est survenue"}), 500

#Evalution de reponse
#Evaluation de reponse
#!pip install spacy
#!python -m spacy download fr_core_news_sm

# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")
spell = SpellChecker(language='fr')
def evaluate_eloquence_and_style(text):
    # Analyser le texte avec spaCy
    doc = nlp(text)
    
    # Évaluer la variété du vocabulaire en comptant le nombre de mots uniques
    unique_words = set(token.text for token in doc if token.is_alpha)
    vocabulary_variety_score = len(unique_words) / len(doc) if len(doc) > 0 else 0
    
    # Évaluer la structure des phrases en analysant la longueur moyenne des phrases
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if len(sentence_lengths) > 0 else 0
    
    # Évaluer la cohérence en analysant la répétition des idées et des thèmes
    word_frequencies = {}
    for token in doc:
        if token.is_alpha:
            word_frequencies[token.text] = word_frequencies.get(token.text, 0) + 1
    coherence_score = sum(freq for freq in word_frequencies.values() if freq > 1) / len(doc) if len(doc) > 0 else 0
    
    # Calculer le score d'éloquence et de style en combinant les scores précédents
    eloquence_and_style_score = (vocabulary_variety_score + avg_sentence_length + coherence_score) / 3
    
    # Inverser le score pour rendre le résultat plus intuitif
    eloquence_and_style_score = 1 - eloquence_and_style_score
    
    return eloquence_and_style_score
def evaluate_grammar_and_spelling(text):
    # Analyser le texte avec spaCy
    doc = nlp(text)
    
    # Vérifier l'orthographe
    corrected_text = " ".join([spell.correction(token.text) for token in doc if token.text.isalpha()])
    spelling_errors = sum(1 for x, y in zip(text.split(), corrected_text.split()) if x.lower() != y.lower())
    spelling_score = 1 - (spelling_errors / len(text.split())) if len(text.split()) > 0 else 0
    
    # Vérifier la grammaire (en utilisant les tags POS)
    grammar_errors = sum(1 for token in doc if token.tag_ not in ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'])
    grammar_score = 1 - (grammar_errors / len(doc)) if len(doc) > 0 else 0
    
    return spelling_score, grammar_score
def evaluate_engagement_and_creativity(text):
    # Analyser le texte avec spaCy
    doc = nlp(text)
    
    # Évaluer l'engagement en analysant la pertinence et la profondeur des idées et des perspectives
    # Évaluer la créativité en analysant l'originalité et l'innovation des solutions proposées
    
    engagement_score =1- sum(1 for token in doc if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']) / len(doc) if len(doc) > 0 else 0
    creativity_score =1- sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in ['le', 'la', 'les', 'de', 'du', 'des']) / len(doc) if len(doc) > 0 else 0
    
    return engagement_score, creativity_score
def evaluate_coherence_and_structure(text):
    # Analyser le texte avec spaCy
    doc = nlp(text)
    
    # Analyser la structure des phrases et l'utilisation des connecteurs logiques
    # Calculer le nombre de tokens qui sont des racines (ROOT) ou des dépendants de type conjonction (conj) ou complément circonstanciel (ccomp)
    coherence_tokens = sum(1 for token in doc if token.dep_ in ['ROOT', 'conj', 'ccomp'])
    
    # Calculer le score de cohérence en divisant le nombre de tokens cohérents par la longueur totale du texte
    coherence_score = coherence_tokens / len(doc) if len(doc) > 0 else 0
    
    # Inverser le score pour que plus le score soit proche de 1, plus la cohérence soit forte
    inverted_coherence_score = 1 - coherence_score
    
    return inverted_coherence_score
def evaluate_clarity(text):
    # Analyser le texte avec spaCy
    doc = nlp(text)
    
    # Calculer la longueur moyenne des phrases
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # Calculer le nombre moyen de mots par phrase
    word_counts = [len(sent.text.split()) for sent in doc.sents]
    avg_word_count = sum(word_counts) / len(word_counts)
    
    # Calculer la longueur moyenne des mots
    avg_word_length = sum(len(token) for token in doc if token.is_alpha) / len([token for token in doc if token.is_alpha])
    
    # Calculer la clarté en fonction de la longueur des phrases, du nombre de mots par phrase et de la longueur des mots
    # Inverser la formule pour que plus le score soit proche de 1, plus la réponse soit considérée comme claire
    normalized_clarity_score = 1 / (avg_sentence_length * avg_word_count * avg_word_length)
    print(normalized_clarity_score)
    clarity_score = 1 - normalized_clarity_score
    print(clarity_score)
    return clarity_score
@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer_api():
    # Extraire la réponse de l'utilisateur depuis la requête POST
    user_answer = request.form['user_answer']
    expected_keywords = request.form['expected_keywords'].split(',')
    print(user_answer)
    print(expected_keywords[0])
    doc = nlp(user_answer)
    
    # Extraire les mots-clés pertinents de la réponse
    user_keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and token.text.lower() not in ['le', 'la', 'les', 'de', 'du', 'des']]
    
    # Calculer la pertinence de la réponse
    relevance_score = len(set(user_keywords) & set(expected_keywords)) / len(set(expected_keywords))
    
    # Évaluer la clarté de la réponse
    clarity_score = evaluate_clarity(user_answer)
    
    # Évaluer la concision de la réponse
    conciseness_score = 1 / len(doc)
    
    # Évaluer la précision de la réponse
    coherence_score = evaluate_coherence_and_structure(user_answer)
    precision_score = 1 if all(keyword in user_keywords for keyword in expected_keywords) else 0
    spelling_score, grammar_score = evaluate_grammar_and_spelling(user_answer)
    engagement_score, creativity_score = evaluate_engagement_and_creativity(user_answer)
    eloquence_score = evaluate_eloquence_and_style(user_answer)

    
    # Calculer le score de qualité global
    quality_score = ( clarity_score +  precision_score   +  coherence_score) / 3
    
    # Retourner les scores évalués sous forme de texte JSON
    response = {
        'relevance': relevance_score,
        'clarity': clarity_score,
        'conciseness': conciseness_score,
        'precision': precision_score,
        'spelling': spelling_score,
        'grammar': grammar_score,
        'coherence': coherence_score,
        'engagement': engagement_score,
        'creativity': creativity_score,
        'eloquence_score': eloquence_score,
        'quality': quality_score
    }
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)