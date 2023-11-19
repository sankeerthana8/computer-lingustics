import stanza
import spacy_stanza
from spacy.matcher import DependencyMatcher
from spacy import displacy
import spacy
stanza.download("en")
nlp = spacy_stanza.load_pipeline("en")

doc = nlp("The doctor gave the spotted lemon to the doctor")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)
displacy.render(nlp("The doctor gave the spotted lemon to the doctor"), jupyter=True)

stanza.download("en")
nlp1 = spacy.load("en_core_web_sm")

doc1 = nlp1("The doctor gave the spotted lemon to the doctor")
for token in doc1:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc1.ents)
displacy.render(nlp1("The doctor gave the spotted lemon to the doctor"), jupyter=True)
