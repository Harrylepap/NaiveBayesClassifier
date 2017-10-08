# Naive Bayes Classifier pour Machine Learning

[![N|Solid](http://www.eskimoz.fr/wp-content/uploads/2017/05/impact-du-Machine-Learning-sur-le-SEO.png)](https://www.facebook.com/rakotondratompobakoharry)

Naive Bayes Classifier est un algorithme populaire en Machine Learning. C’est un algorithme du Supervised Learning utilisé pour la classification. Il est particulièrement utile pour les problématiques de classification de texte. Un exemple d’utilisation du Naive Bayes est celui du filtre anti-spam. 
On mettra en place un « SPAM Filter » en utilisant le Naive Bayes Classifier. Notre classifieur se basera sur Python et sa librairie de Machine Learning : « Sickit Learn ».

L’algorithme prendra en entrée un e-mail et nous indiquera s’il s’agit d’un mail ou non.

Comme plan, nous avons

  - Présentation des données
  - Chargement des données
  - Préparation du classifieur Naïve Bayes
  - Entrainement du Naive Bayes Classifier
  - Tester le Spam Filter
  - Piste d’améliorations
  - Conlusion

### Présentation des données
Naive Bayes est un algorithme d’apprentissage supervisé. De ce fait, Il faudra qu’on lui fournisse une base de données d’apprentissage supervisé.

Notre jeu d’apprentissage sera une collection de mails. Pour chacun d’eux, on indiquera s’il s’agit d’un SPAM ou non.

| Données d’entrée | Label |
| ------ | ------ |
| 1er mail |Spam |
| 2eme mail | Non spam |
| 3eme mail | Non spam |
|... | ....|
Notre Training Set est composé de :
  - 500 mails Spam
  - 2500 mails Non spam (Ham)



### Chargement des données
Tout d’abord, on chargera nos mails en utilisant les modules « OS » et « IO » de Python. Voici le snippet de code de lecture et chargement des mails :

```python
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
```

A l’exécution de ce code, on chargera 3000 mails (ce qui constitue notre Training Set).
> Pensez à modifier le chemin vers le répertoire contenant les mails lors de l’exécution du script sur votre machine.

### Préparation du classifieur Naïve Bayes
Je vais illustré comment opère cet algorithme en classifiant des fruits en fonction de leurs caractéristiques. On calcul une probabilité conditionnelle pour savoir s’il s’agit d’une banane, d’une orange ou un autre fruit. La probabilité conditionnelle est calculée en se basant sur la cardinalité/proportion de chaque caractéristique du fruit dans notre jeu de données.
Pour la classification de mails en SPAM/Non Spam, c’est la même logique qui s’applique. En effet, les mails SPAM contiennent généralement une proportion élevée de mots comme « Gagner, Gratuit,…etc » alors que dans un mail normal, ce ne sera pas le cas.
>Pour classifier nos emails avec Naïve Bayes, on calcule le nombre d’occurrence des mots dans chaque mail.

On doit, pour chaque mail, comptabiliser le nombre d’occurrence des mots. Cela permettra de calculer la probabilité conditionnelle comme dans l’exemple de classification des fruits.

Python nous simplifie la vie en nous proposant une fonction qui effectue ce calcul. Il s’agit de la méthode fit_transform de la classe countVectorizer.

```python
vecteur = CountVectorizer()
counts = vecteur.fit_transform(matrice['X'].values)
```
La variable counts contiendra une matrix contenant, pour chaque document, le nombre d’occurences de chaque mots

### Entrainement du Naive Bayes Classifier


Tous les ingrédients sont là, il nous reste qu’a entraîner notre filtre anti-spam. Pour ce faire, on lui filera les mails 3000 mails qu’on a chargé auparavant. Pour chaque mail que le classifieur lira, on lui indiquera s’il s’agit de SPAM ou non.

Sickit_learn nous propose la classe MultinomialNB (NB pour Naïve Bayes), qui fera tous le travail pour nous.

```python
classifier = MultinomialNB()
targets = matrice['Y'].values
classifier.fit(counts, targets)
```
Note :

- Pour être plus précis, on donne à MultinomialNB() la version « codifiée » des mails qu’on a calculé avec CountVectorizer
- Pour chaque mail, on spécifie s’il s’agit de SPAM ou non (voir l’appel classifier.fit(counts, targets))

Et voilà ! 
### Tester le Spam Filter

Il nous reste plus qu’à tester notre filtre anti-spam. Certes, il n’est pas de la tremple que celui de Google, mais, regardons ce qu’il a dans le ventre.
-Vu que les mails de notre Training Set sont en anglais et quelques mail en Malagasy, on lui donnera un tableau de mail à travailler
```python
TableauDePhrase = ['Free Viagra now!!!', "Bonjour Sandrine, comment ca va? je t'aime", "Ndao hilely fory be"]
compteurVecteur = vecteur.transform(TableauDePhrase)
predictions = classifier.predict(compteurVecteur)
```

La sortie de l'application donnera :
```batch
Free Viagra now!!!  =>  SPAM
Bonjour Sandrine, comment ca va? je t'aime  =>  HAM | Message normale
Ndao hilely fory be  =>  SPAM
```

### Piste d’améliorations

Le Spam Filter qu’on vient d’implémenter est assez rudimentaire. En effet, par défaut, countVectorizer comptabilise les occurrences de chaque mots. Par ailleurs, les mots « Apple » et « Apples » peuvent être considérés différents. La même histoire pour le respect de la casse (par exemple GRATUIT et gratuit).

Pour mieux optimiser l’algorithme, on peut appliquer quelques techniques de Text Analysis. La documentation de Python en [explique le principe avec un des exemples illustratifs.](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform)

### Conclusion
Vous venez d’implémenter votre premier Spam Filter. Tentez d’entraîner ce filtre sur vos mails Français et partagez vos retrouvailles en commentaire 😉