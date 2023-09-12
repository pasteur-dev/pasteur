import pasteur

# Inject pasteur version
with open("requirements.in", "r") as f:
    reqs = f.read()

reqs = reqs.replace("pasteur[opt,test]", f"pasteur[opt,test]~={pasteur.version}")

with open("requirements.in", "w") as f:
    f.write(reqs)
