Classificação de Não Pagamento do Cartão de Crédito
===================================================

O problema consiste em classificar corretamente se um dado cliente vai ou não pagar a fatura do seu cartão de crédito no mês seguinte. 

Setup
-----

Crie um ambiente virtual e ative-o:

    $ make create_environment
    $ workon credit-card-default

Instale os pacotes necessários:

    $ make requirements



Base de Dados
-------------

Este projeto utiliza a base de dados ["Default of Credit Card Clients Dataset"](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) disponibilizada pela UCI Machine Learning no site do Kaggle. A base contém 25 variáveis. Abaixo é transcrita a descrição dada no Kaggle.
  
> **ID:** ID of each client    
> **LIMIT_BAL:** Amount of given credit in NT dollars (includes individual and family/supplementary credit  
> **SEX:** Gender (1=male, 2=female)  
> **EDUCATION:** (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)  
> **MARRIAGE:** Marital status (1=married, 2=single, 3=others)  
> **AGE:** Age in years  
> **PAY_0:** Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)  
> **PAY_2:** Repayment status in August, 2005 (scale same as above)  
> **PAY_3:** Repayment status in July, 2005 (scale same as above)  
> **PAY_4:** Repayment status in June, 2005 (scale same as above)  
> **PAY_5:** Repayment status in May, 2005 (scale same as above)  
> **PAY_6:** Repayment status in April, 2005 (scale same as above)  
> **BILL_AMT1:** Amount of bill statement in September, 2005 (NT dollar)  
> **BILL_AMT2:** Amount of bill statement in August, 2005 (NT dollar)  
> **BILL_AMT3:** Amount of bill statement in July, 2005 (NT dollar)  
> **BILL_AMT4:** Amount of bill statement in June, 2005 (NT dollar)  
> **BILL_AMT5:** Amount of bill statement in May, 2005 (NT dollar)  
> **BILL_AMT6:** Amount of bill statement in April, 2005 (NT dollar)  
> **PAY_AMT1:** Amount of previous payment in September, 2005 (NT dollar)  
> **PAY_AMT2:** Amount of previous payment in August, 2005 (NT dollar)  
> **PAY_AMT3:** Amount of previous payment in July, 2005 (NT dollar)  
> **PAY_AMT4:** Amount of previous payment in June, 2005 (NT dollar)  
> **PAY_AMT5:** Amount of previous payment in May, 2005 (NT dollar)  
> **PAY_AMT6:** Amount of previous payment in April, 2005 (NT dollar)  
> **default.payment.next.month:** Default payment (1=yes, 0=no)


Dados tratados
-------------

Este projeto utiliza a base de dados ["Default of Credit Card Clients Dataset"](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) disponibilizada pela UCI Machine Learning no site do Kaggle. A base contém 25 variáveis. Abaixo é transcrita a descrição dada no Kaggle.
  
> **ID:** ID of each client    
> **LIMIT_BAL:** Amount of given credit in NT dollars (includes individual and family/supplementary credit  
> **SEX:** Gender (0=male, 1=female)  
> **ED_1:** (education type 1=graduate school 0=otherwise)
> **ED_2:** (education type 1=university 0=otherwise)
> **ED_3:** (education type 1=high school 0=otherwise)
> **ED_4:** (education type 1=others 0=otherwise)
> **MARRIAGE:** Marital status (1=married, 0 = not maried(others))  
> **AGE:** Age in years normalized to mean 0 and variance 1
> **PAY_0:** Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)  
> **PAY_2:** Repayment status in August, 2005 (scale same as above)  
> **PAY_3:** Repayment status in July, 2005 (scale same as above)  
> **PAY_4:** Repayment status in June, 2005 (scale same as above)  
> **PAY_5:** Repayment status in May, 2005 (scale same as above)  
> **PAY_6:** Repayment status in April, 2005 (scale same as above)  
> **BILL_AMT1:** Amount of bill statement in September, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **BILL_AMT2:** Amount of bill statement in August, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **BILL_AMT3:** Amount of bill statement in July, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **BILL_AMT4:** Amount of bill statement in June, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **BILL_AMT5:** Amount of bill statement in May, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **BILL_AMT6:** Amount of bill statement in April, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **PAY_AMT1:** Amount of previous payment in September, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **PAY_AMT2:** Amount of previous payment in August, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **PAY_AMT3:** Amount of previous payment in July, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **PAY_AMT4:** Amount of previous payment in June, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **PAY_AMT5:** Amount of previous payment in May, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **PAY_AMT6:** Amount of previous payment in April, 2005 (NT dollar)  normalized to mean 0 and variance 1
> **default.payment.next.month:** Default payment (1=yes, 0=no)

<p><small>Projeto baseado no template <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
