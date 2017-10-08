import os, io 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#FONCTIONS
def lectureContenuMailSansHeader(nom):
    contenueMail = False
    ligne = []
    descriptionFichier = io.open(nom, 'r', encoding='latin1')
    for line in descriptionFichier:
        if contenueMail:
            ligne.append(line)
        elif line == '\n':
            contenueMail = True
        message = '\n'.join(ligne)
    descriptionFichier.close()
    return message


def lectureMailDossier(chemin, classification):   
    os.chdir(chemin)
    fichiers = os.listdir(chemin)
    for fichier in fichiers:
        message = lectureContenuMailSansHeader(fichier)
        X.append(message)
        Y.append(classification)


#DATA
## dossier
dossierHam =  r'C:\Users\Harrylepap\PycharmProjects\anti-spam-machine-learning\emails\ham'
dossierSpam =  r'C:\Users\Harrylepap\PycharmProjects\anti-spam-machine-learning\emails\spam'

## type de mail
spam_type = "SPAM"
ham_type  = "HAM | Message normale"

## initialisation
i=0
##les tableaux X et Y seront de la meme taille et ordonnes
### represente l'input Data (ici les mails)
X = []
### indique s'il s'agit d'un mail ou non
### etiquettes (labels) pour la matrice
Y = []


#TRAITEMENTS
lectureMailDossier(dossierHam, ham_type)
lectureMailDossier(dossierSpam, spam_type)

matrice = pd.DataFrame({'X': X, 'Y': Y})

vecteur = CountVectorizer()
counts = vecteur.fit_transform(matrice['X'].values)

classifier = MultinomialNB()
targets = matrice['Y'].values
classifier.fit(counts, targets)


TableauDePhrase = ['Free Viagra now!!!', "Bonjour Sandrine, comment ca va? je t'aime", "Ndao hilely fory be"]
compteurVecteur = vecteur.transform(TableauDePhrase)
predictions = classifier.predict(compteurVecteur)

nombres = len(TableauDePhrase)
nombres = int(nombres)

#SORTIE
for message in TableauDePhrase:
    print(message, " => ", predictions[i])
    i+=1