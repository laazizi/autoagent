"""Cache de prompt (0.19.0) — comptabilité et activation.

Un agent renvoie tout le transcript à chaque tour : le prompt système et les
schémas d'outils repartent identiques à chaque étape. Les fournisseurs savent
servir ce préfixe depuis un cache — mais chacun le COMPTE différemment, et
c'est là qu'un bug silencieux se loge.

Ces tests verrouillent deux choses :

  1. La NORMALISATION. Le même fait — « 1 000 jetons d'entrée dont 800 servis
     par le cache » — arrive sous trois formes de fil incompatibles. Après
     traversée de la frontière, il doit produire un `TokenUsage` identique,
     sinon `token_budget` compte faux dès que le cache mord.

  2. L'ACTIVATION. Anthropic est le seul à exiger un marqueur explicite ; sans
     `cache_prompt`, le payload doit rester exactement ce qu'il était.
"""

from autoagent import Message, ModelConfig
from autoagent.providers.anthropic import AnthropicProvider
from autoagent.providers.anthropic import _usage_from as anthropic_usage
from autoagent.providers.gemini import _usage_from as gemini_usage
from autoagent.providers.openai import _usage_from as openai_usage
from autoagent.schema import LLMRequest, LLMResponse, TokenUsage


class TestNormalisationEntreFournisseurs:
    """Le MÊME fait, dit de quatre façons, doit rendre le même compte."""

    ATTENDU = (1000, 50, 800)          # entrée totale, sortie, dont cache

    def _verifier(self, usage: TokenUsage) -> None:
        entree, sortie, cache = self.ATTENDU
        assert usage.input_tokens == entree, "l'entrée totale n'est pas normalisée"
        assert usage.output_tokens == sortie
        assert usage.cached_tokens == cache
        # Le cache est un SOUS-ENSEMBLE de l'entrée : il ne doit jamais gonfler
        # le total, sinon on facturerait deux fois les mêmes jetons.
        assert usage.total_tokens == entree + sortie

    def test_openai(self) -> None:
        self._verifier(openai_usage({
            "prompt_tokens": 1000, "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 800},
        }))

    def test_deepseek_qui_nomme_le_champ_autrement(self) -> None:
        self._verifier(openai_usage({
            "prompt_tokens": 1000, "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200,
        }))

    def test_gemini(self) -> None:
        self._verifier(gemini_usage({
            "promptTokenCount": 1000, "candidatesTokenCount": 50,
            "totalTokenCount": 1050, "cachedContentTokenCount": 800,
        }))

    def test_anthropic_compte_le_cache_A_COTE_de_l_entree(self) -> None:
        """Le cas qui justifie tout ce fichier.

        Anthropic rend `input_tokens` = l'entrée NON mise en cache. Recopier
        tel quel donnerait 200 au lieu de 1 000 : `token_budget` croirait le
        run cinq fois moins cher qu'il n'est.
        """
        self._verifier(anthropic_usage({
            "input_tokens": 200, "cache_read_input_tokens": 800,
            "output_tokens": 50,
        }))


class TestAnthropicEcritureDuCache:
    def test_l_ecriture_compte_aussi_dans_l_entree(self) -> None:
        """Premier appel : le cache est ÉCRIT, pas lu. Ça se paie, et plus cher.

        L'écriture doit donc entrer dans le total, mais ne jamais être comptée
        comme une économie.
        """
        usage = anthropic_usage({
            "input_tokens": 200, "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 300, "output_tokens": 50,
        })
        assert usage.input_tokens == 1000
        assert usage.cached_tokens == 300, "seule la LECTURE est une économie"

    def test_sans_cache_le_compte_est_inchange(self) -> None:
        usage = anthropic_usage({"input_tokens": 700, "output_tokens": 30})
        assert (usage.input_tokens, usage.output_tokens) == (700, 30)
        assert usage.cached_tokens is None, "aucun zéro inventé"


class TestTauxDeCache:
    def test_calcul(self) -> None:
        assert TokenUsage(input_tokens=1000, cached_tokens=800).cache_hit_ratio == 0.8

    def test_inconnu_quand_le_fournisseur_ne_dit_rien(self) -> None:
        assert TokenUsage(input_tokens=1000).cache_hit_ratio is None

    def test_pas_de_division_par_zero(self) -> None:
        assert TokenUsage(input_tokens=0, cached_tokens=0).cache_hit_ratio is None


class TestActivationAnthropic:
    def _payload(self, cache: bool) -> dict:
        fournisseur = AnthropicProvider(ModelConfig(
            provider="anthropic", model="claude-sonnet-4-5",
            api_key="x", cache_prompt=cache))
        return fournisseur._build_payload(LLMRequest(messages=[
            Message(role="system", content="Tu es un assistant."),
            Message(role="user", content="Bonjour"),
        ]))

    def test_eteint_le_payload_ne_bouge_pas(self) -> None:
        assert self._payload(cache=False)["system"] == "Tu es un assistant."

    def test_allume_le_bloc_systeme_porte_le_marqueur(self) -> None:
        systeme = self._payload(cache=True)["system"]
        assert isinstance(systeme, list) and len(systeme) == 1
        assert systeme[0]["text"] == "Tu es un assistant."
        assert systeme[0]["cache_control"] == {"type": "ephemeral"}

    def test_le_defaut_est_eteint(self) -> None:
        """Opt-in : chez Anthropic l'écriture du cache coûte plus cher que
        l'entrée normale, donc l'activer par défaut ferait perdre de l'argent
        sur les préfixes courts ou utilisés une seule fois."""
        assert ModelConfig(provider="anthropic", model="m").cache_prompt is False


class TestSerialisation:
    def test_aller_retour(self) -> None:
        avant = LLMResponse(content="ok", usage=TokenUsage(
            input_tokens=1000, output_tokens=50, cached_tokens=800))
        apres = LLMResponse.from_dict(avant.to_dict())
        assert apres.usage.cached_tokens == 800

    def test_un_enregistrement_anterieur_se_relit(self) -> None:
        """Un fixture de rejeu écrit avant 0.19.0 n'a pas la clé : il doit se
        relire en rendant None, jamais en plantant ni en inventant un zéro."""
        ancien = {"content": "ok", "usage": {"input_tokens": 10, "output_tokens": 2}}
        assert LLMResponse.from_dict(ancien).usage.cached_tokens is None


class TestAgregationSurUnRun:
    """« 0 servi par le cache » est une mesure ; « rien rapporté » est une absence.

    Bug trouvé en écrivant ces tests : l'agrégation faisait `spent_cached or None`,
    ce qui transformait un zéro MESURÉ en « inconnu » — donc rendait impossible de
    savoir si activer le cache avait servi à quelque chose.
    """

    def _usage_du_run(self, cached):
        from autoagent import Agent
        from autoagent.providers.base import LLMProvider

        class Fournisseur(LLMProvider):
            def __init__(self) -> None:
                super().__init__(ModelConfig(provider="f", model="f", api_key="x"))

            def complete(self, request):
                return LLMResponse(content="fini", usage=TokenUsage(
                    input_tokens=1000, output_tokens=50, cached_tokens=cached))

        return Agent(Fournisseur()).run("x").usage

    def test_fournisseur_muet_rend_inconnu(self) -> None:
        assert self._usage_du_run(None).cached_tokens is None

    def test_zero_mesure_reste_zero(self) -> None:
        usage = self._usage_du_run(0)
        assert usage.cached_tokens == 0, "un zéro mesuré ne doit pas devenir « inconnu »"
        assert usage.cache_hit_ratio == 0.0

    def test_le_compte_remonte_jusqu_au_resultat(self) -> None:
        usage = self._usage_du_run(800)
        assert usage.cached_tokens == 800
        assert usage.cache_hit_ratio == 0.8
