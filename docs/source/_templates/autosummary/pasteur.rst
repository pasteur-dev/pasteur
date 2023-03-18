{{ fullname | escape | underline }}

.. rubric:: Description

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :template: autosummary/class.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
      :template: autosummary/class.rst
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% set pasteur_attr = ["pasteur.attribute","pasteur.hierarchy","pasteur.table"] %}
{% set pasteur_modules = ["pasteur.module", "pasteur.dataset","pasteur.view","pasteur.transform","pasteur.encode","pasteur.synth","pasteur.metric"] %}
{% set pasteur_misc = ["pasteur.kedro","pasteur.utils","pasteur.extras","pasteur.cli"] %}

.. rubric:: Module-System Modules

.. autosummary::
   :toctree:
   :template: autosummary/module.rst
   :recursive:
{% for item in pasteur_modules %}
   {{ item }}
{%- endfor %}

.. rubric:: Transformation-Related Modules

.. autosummary::
   :toctree:
   :template: autosummary/module.rst
   :recursive:
{% for item in pasteur_attr %}
   {{ item }}
{%- endfor %}

{% block modules %}
{% if modules|length > pasteur_modules|length + pasteur_misc|length + pasteur_attr|length %}
.. rubric:: Other Modules

.. autosummary::
   :toctree:
   :template: autosummary/module.rst
   :recursive:
{% for item in modules %}
   {% if item not in pasteur_modules and item not in pasteur_attr and item not in pasteur_misc %}
      {{ item }}
   {% endif %}
{%- endfor %}
{% endif %}
{% endblock %}

.. rubric:: Miscellaneous Modules

.. autosummary::
   :toctree:
   :template: autosummary/module.rst
   :recursive:
{% for item in pasteur_misc %}
   {{ item }}
{%- endfor %}
