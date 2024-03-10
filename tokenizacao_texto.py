# Exercicio PP 1.1

# Imports das Bibliotecas
import nltk
from nltk import pos_tag

# Downloads
nltk.download('punkt')
nltk.download('mac_morpho')
nltk.download('averaged_perceptron_tagger')


# Texto 2 literario - Obra: O Pacto maldito e outras histórias de morte | Autor: José Claudio da Silva
texto = """
Tudo começou numa quarta-feira quente de janeiro. Era mês de colocar as leituras em dia. Esquecer o mundo lá fora e se embrenhar durante trinta dias no mundo imaginário dos livros! Brammmm! A porta da sala bateu violentamente contra a parede.
Abriram com uma bomba poderosa: o pontapé. Os bárbaros chegaram. - Pai, quero jogar videogame! - Tio, quero Nescau! - Tio, quero guaraná! - Pai, quero bolacha! Todas falavam ao mesmo tempo. Exigindo seus direitos. Eram seis anjinhos: Daniel e André, filhos. Maurício, Vinícius, Lucas, Mayra, a única menina bárbara, sobrinhos. - Mães irresponsáveis. Largaram essas crianças neste lugar minúsculo e foram passear no shopping.
Como vou ler com tanto barulho!? Enquanto isso, as crianças começaram uma guerra de almofadas no quarto do casal. - Parem com esse barulho. Venham aqui. Empurrando, chutando, esmurrando, xingando uns aos outros, chegaram na porta da saleta. Todas tentavam passar, ao mesmo tempo, pela porta. - Silêncio!!! Sentem-se na mesa. Não em cima da mesa! Vocês vão quebrar o vidro e sua mãe me mata.
Pelo amor de Deus, sentem-se nas cadeiras! Tudo bem. Agora silêncio! Só eu falo! O que vocês querem fazer além de destruir o apartamento e me deixar louco? Gritando, falavam ao mesmo tempo: - Quero ir ao Play Center, ao zoológico, Cidade da Criança, ao Parque da Mônica... - Nem pensar! Eu não sou louco para ficar andando com seis crianças, cheias de energia, em parques enormes. 
Que tal um programa diferente? - Qual?- Acampar!!! - Acampar??? - É. Vocês nunca acamparam? Mas não vamos a um camping cheio de confortos. Vamos acampar numa mata, sozinhos. - Igual aos escoteiros, tio? - Mais ou menos, Lucas. - Vai ser legal, tio? - Garanto que vocês vão gostar. Vamos fazer muitas coisas. Será divertido. - Para onde vamos, tio? - perguntou Mayra. - Você não vai - gritou Daniel. - Acampamento é só para homens - completou Vinícius. Mayra mostrou a língua para os dois e disse: - Bobões. Eu também vou, tio? - Claro! - Alguém vai ter de fazer a comida e lavar a louça - resmungou Lucas. - Quero que vocês peguem as mochilas da escola e tirem os materiais de dentro e... - Que mochilas, tio? - perguntou Lucas - a minha não presta para mais nada. - Nem a minha! - Nem a minha! - Posso imaginar o estado em que elas se encontram. Vou sair para comprar mochilas, mantimentos e alguns utensílios para acampar. Por favor, não destruam o apartamento enquanto eu estiver fora. 
Ao voltar do supermercado, uma gritaria infernal o recebeu. Parecia que o mundo estava se acabando num terrível cataclismo. Era como se um bando de dentistas com brocas e enfermeiras com injeções tivessem invadido o apartamento e perseguisse todas as crianças, que fugiam desesperadas, apavoradas, aos berros derrubando tudo que encontravam pelo caminho. Um verdadeiro campo de batalha final. - Crianças, parem com esse inferno e venham aqui! Todas correram para a cozinha se atropelando e derrubando coisas pelo caminho. -Silêncio! Já comprei tudo o que precisaremos para acampar. 
Ficaremos uma semana na mata, sem shopping e MacDonalds. Vamos arrumar as coisas dentro das mochilas que eu comprei para vocês. Amanhã bem cedo, partiremos para nossa aventura. - Minha mãe não vai me deixar ir, tio.-Vai sim, Maurício. Suas mães vão adorar ficar uma semana longe de vocês nas férias. Agora cada um pegue uma mochila e coloque as coisas que eu comprei. Tem um conjunto para cada um. As mochilas são iguais, mas sempre têm dois querendo a mesma. - Parem com essa briga!
"""

# Texto escolhido (POS corretas)
texto2 = """
É a única rua de Lisboa com o trânsito ao contrário, sabias?. Não sabia e é por isso que atravesso em frente dos carros tantas vezes. Eles apitam e eu dou uma corrida. "O trânsito é ao contrário, não me posso esquecer".
A mala esta pesada, cheia de papéis rabiscados que demoro semanas a deitar fora. Nunca se sabe. Nas mãos há sempre um casaco, um cachecol, este inverno está uma treta, nem sequer faz frio. Já não basta a eterna ausência de neve.
Às vezes quero calar a cabeça. Ficar com o olhar preso no nada, simplesmente. Nasci com uma estranha deficiência de não conseguir não pensar em nada. Deve ser por isso que os meus sonhos não se entendem, colagens sobrepostas de pensamentos a mil à hora.
O mundo inteiro na minha cabeça. A China, o Saramago, o bolo de chocolate e o senhor que passeia o cão à noite. O café da máquina, as saudades da amiga, o verso do poema, os olhos ansiosos do namorado. Israel, EUA, Bruxelas. O Tratado de Lisboa e a série de televisão que deixei de ver. Que será feito da Rory?
Desligar a cabeça.
Como? Talvez o David Lynch saiba, ele medita, ele vai criar uma universidade da meditação. Curioso...
Mas agora o mundo todo são palavras a esferográfica num bloco azul. "Por favor, não diga mais, porque eu não sei escolher". Na minha cabeça cabe o mundo. E o mundo tem que caber em 2000 caracteres. 
"""

# Tokenização por sentenças
sentencas = nltk.sent_tokenize(texto, language='portuguese')
#sentencas = nltk.sent_tokenize(texto2, language='portuguese')

# Tokenização por palavras
palavras = nltk.word_tokenize(texto, language='portuguese')
#palavras = nltk.word_tokenize(texto2, language='portuguese')

# Exibindo os resultados
print("Tokenização por Sentenças:")
print(sentencas)

print("\n Tokenização por Palavras:")
print(palavras)

# POS Tagging
tags_pos = pos_tag(palavras)

for palavra, tag in tags_pos[:5]:
    print(f"\nPalavra: {palavra}, POS Tag: {tag}")