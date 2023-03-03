import pasteur

# Inject pasteur version
with open('src/requirements.txt', "r") as f:
    reqs = f.read()

reqs = reqs.replace("pasteur[opt,test,docs]", f"pasteur[opt,test,docs]~={pasteur.version}")

with open('src/requirements.txt', "w") as f:
    f.write(reqs)