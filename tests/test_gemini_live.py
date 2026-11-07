"""Le fournisseur Gemini Live (`autoagent.providers.gemini_live`) : la voix en continu, sous la boucle d'agent.

Le serveur de test (`tests/_fake_ws.py`) reproduit ce qui a été OBSERVÉ sur l'API réelle le 2026-11-07
(modèle `gemini-3.8-live`), pas ce que dit la documentation :

* le JSON du serveur arrive dans des trames BINAIRES ;
* la configuration valide est `setup.generationConfig.responseModalities` — la forme de la page officielle
  (`setup.responseModalities`) est REFUSÉE par le serveur (fermeture 1007) : le faux serveur la refuse aussi ;
* un `toolCall` est suivi IMMÉDIATEMENT de `generationComplete` puis `turnComplete`, qui terminent le tour qui a
  produit l'appel, avant même que l'outil ait tourné ; la vraie réponse (audio) vient après la `toolResponse`, et
  son `turnComplete` n'arrive qu'environ la durée de l'audio plus tard, avec `usageMetadata`.

Un fournisseur qui prend le premier `turnComplete` pour la fin de la réponse rend une réponse VIDE après chaque
appel d'outil : c'est le défaut que ces tests existent pour attraper. Le point le plus important de la série : un
outil demandé par Gemini passe par la MÊME porte que les autres (politique, gardes) — la boucle ne change pas.
"""

from __future__ import annotations

import base64
import json
import threading
import time

import pytest

from autoagent import Agent
from autoagent.errors import ProviderError
from autoagent.providers import create_provider
from autoagent.providers.gemini_live import GeminiLiveProvider
from autoagent.schema import ModelConfig
from tests._fake_ws import FakeWSServer

CLE = "CLE-SECRETE-1234"


# ── le faux serveur Live ─────────────────────────────────────────────────────────────
def _json(conn, objet):
    conn.binaire(json.dumps(objet).encode("utf-8"))        # le vrai serveur parle JSON en trames binaires


def _audio(octets: bytes) -> dict:
    return {"serverContent": {"modelTurn": {"parts": [{"inlineData": {
        "mimeType": "audio/pcm;rate=24000", "data": base64.b64encode(octets).decode()}}]}}}


def _parle(conn, texte, morceaux=(b"\x01\x02" * 50,), *, usage=None, delai_avant_fin=0.0):
    """Un tour de parole complet tel que l'API le produit : audio, transcription, generationComplete, puis
    turnComplete (+ usageMetadata) un peu plus tard."""
    milieu = len(texte) // 2
    for morceau in morceaux:
        _json(conn, _audio(morceau))
    _json(conn, {"serverContent": {"outputTranscription": {"text": texte[:milieu]}}})
    _json(conn, {"serverContent": {"outputTranscription": {"text": texte[milieu:]}}})
    _json(conn, {"serverContent": {"generationComplete": True}})
    if delai_avant_fin:
        time.sleep(delai_avant_fin)
    fin = {"serverContent": {"turnComplete": True}}
    if usage:
        fin["usageMetadata"] = usage
    _json(conn, fin)


def _appel_outil(conn, nom="avancer", args=None, identifiant="call_42", *, paire_perimee=True):
    """Un `toolCall`, suivi (comme sur l'API réelle) de la paire generationComplete/turnComplete du tour périmé."""
    _json(conn, {"toolCall": {"functionCalls": [{"name": nom, "args": args or {"cm": 10}, "id": identifiant}]}})
    if paire_perimee:
        _json(conn, {"serverContent": {"generationComplete": True}})
        _json(conn, {"serverContent": {"turnComplete": True}, "usageMetadata": {
            "promptTokenCount": 100, "responseTokenCount": 10, "totalTokenCount": 110}})


def _entree_client(conn) -> dict:
    return json.loads(conn.recevoir_texte())


@pytest.fixture
def serveur():
    crees: list[FakeWSServer] = []

    def _creer(scenario):
        vu: dict = {"setups": [], "headers": [], "chemins": [], "entrees": []}

        def script(conn):
            vu["headers"].append(dict(conn.headers))
            vu["chemins"].append(conn.path)
            conn.accept()
            setup = _entree_client(conn)
            vu["setups"].append(setup)
            interne = setup.get("setup", {})
            refus = vu.get("refus")                        # un test peut faire refuser une configuration précise
            raison_refus = refus(setup) if refus else None
            if raison_refus:
                conn.fermer_proprement(1008, raison_refus)
                return
            if "responseModalities" in interne:            # ce que fait le VRAI serveur de la page officielle
                conn.fermer_proprement(1007, "Invalid JSON payload received. Unknown name "
                                             "\"responseModalities\" at 'setup': Cannot find field.")
                return
            _json(conn, {"setupComplete": {}})
            scenario(conn, vu)
            conn.attendre_la_fin()
        s = FakeWSServer(script)
        s.vu = vu
        crees.append(s)
        return s

    yield _creer
    for s in crees:
        s.close()
    for s in crees:
        assert not s.errors, f"le scénario du serveur a échoué : {s.errors[0]!r}"


def _agent(s: FakeWSServer, **options) -> Agent:
    config = ModelConfig(provider="gemini_live", model="gemini-3.8-live", api_key=CLE,
                         base_url=s.url("/live"), timeout=5.0)
    options.setdefault("system_prompt", "Tu es un petit robot.")
    return Agent.from_model_config(config, **options)


# ── la configuration envoyée au serveur ──────────────────────────────────────────────
def test_la_configuration_utilise_la_forme_acceptee_par_le_vrai_serveur(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _parle(conn, "Bonjour.")
    s = serveur(scenario)
    agent = _agent(s)

    @agent.tool
    def avancer(cm: float) -> dict:
        """Fait avancer le robot de `cm` centimètres."""
        return {"ok": True}
    agent.run("Dis bonjour")
    setup = s.vu["setups"][0]["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert "responseModalities" not in setup                           # la forme que le serveur refuse
    assert setup["systemInstruction"]["parts"][0]["text"].startswith("Tu es un petit robot.")
    assert setup["tools"][0]["functionDeclarations"][0]["name"] == "avancer"
    assert setup["inputAudioTranscription"] == {} and setup["outputAudioTranscription"] == {}


def test_la_cle_voyage_dans_un_en_tete_jamais_dans_l_url(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _parle(conn, "Bonjour.")
    s = serveur(scenario)
    _agent(s).run("Dis bonjour")
    assert s.vu["headers"][0]["x-goog-api-key"] == CLE
    assert "key=" not in s.vu["chemins"][0] and CLE not in s.vu["chemins"][0]


def test_une_configuration_refusee_par_le_serveur_remonte_sa_raison_et_n_est_pas_reessayable(serveur):
    def scenario(conn, vu):
        pass
    s = serveur(scenario)
    config = ModelConfig(provider="gemini_live", model="gemini-3.8-live", api_key=CLE, base_url=s.url("/live"),
                         timeout=5.0, extra_body={"setup": {"responseModalities": ["AUDIO"]}})   # la forme refusée
    with pytest.raises(ProviderError) as e:
        Agent.from_model_config(config).run("Dis bonjour")
    assert "Unknown name" in str(e.value) and "1007" in str(e.value)
    assert e.value.retryable is False
    assert CLE not in str(e.value)


def test_une_cle_absente_est_une_erreur_claire_avant_toute_connexion():
    config = ModelConfig(provider="gemini_live", model="m", api_key_env="VARIABLE_QUI_N_EXISTE_PAS_XYZ",
                         base_url="ws://127.0.0.1:1/")
    with pytest.raises(ValueError, match="VARIABLE_QUI_N_EXISTE_PAS_XYZ"):
        Agent.from_model_config(config).run("Dis bonjour")


def test_un_refus_http_de_la_poignee_de_main_ne_revele_pas_la_cle(serveur):
    s = FakeWSServer(lambda conn: conn.refuse("403 Forbidden", b"API key not valid"))
    try:
        config = ModelConfig(provider="gemini_live", model="m", api_key=CLE, base_url=s.url("/live?key=" + CLE),
                             timeout=3.0)
        with pytest.raises(ProviderError) as e:
            Agent.from_model_config(config).run("Dis bonjour")
        assert "403" in str(e.value) and CLE not in str(e.value)
    finally:
        s.close()


def test_le_fournisseur_est_enregistre_sous_deux_noms():
    for nom in ("gemini_live", "gemini-live"):
        assert isinstance(create_provider(ModelConfig(provider=nom, model="gemini-3.8-live", api_key="x")),
                          GeminiLiveProvider)


# ── un tour de parole ────────────────────────────────────────────────────────────────
def test_un_tour_de_parole_rend_la_transcription_et_livre_l_audio_au_fil_de_l_eau(serveur):
    morceaux = (b"\x01\x02" * 40, b"\x03\x04" * 60, b"\x05\x06" * 20)

    def scenario(conn, vu):
        vu["entrees"].append(_entree_client(conn))
        _parle(conn, "Bonjour, humain !", morceaux)
    s = serveur(scenario)
    agent = _agent(s)
    audio: list[bytes] = []
    agent.provider.on_audio = audio.append
    resultat = agent.run("Dis bonjour")
    assert resultat.output == "Bonjour, humain !"
    assert audio == list(morceaux)                                      # l'audio sort tel quel, dans l'ordre
    assert s.vu["entrees"] == [{"realtimeInput": {"text": "Dis bonjour"}}]


def test_la_consommation_de_jetons_vient_du_turnComplete_pensees_comprises(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _parle(conn, "Salut.", usage={"promptTokenCount": 569, "responseTokenCount": 67, "totalTokenCount": 636,
                                      "thoughtsTokenCount": 71})
    s = serveur(scenario)
    resultat = _agent(s).run("Dis salut")
    assert resultat.usage.input_tokens == 569
    assert resultat.usage.output_tokens == 67 + 71                      # la sortie facturée, pensées comprises (cf. Gemini REST)
    assert resultat.usage.total_tokens == 569 + 67 + 71


def test_le_run_attend_le_turnComplete_pour_avoir_la_consommation_meme_s_il_arrive_apres_l_audio(serveur):
    """Sur l'API réelle, turnComplete suit generationComplete d'environ la durée de l'audio."""
    def scenario(conn, vu):
        _entree_client(conn)
        _parle(conn, "Salut.", usage={"promptTokenCount": 10, "responseTokenCount": 5, "totalTokenCount": 15},
               delai_avant_fin=0.4)
    s = serveur(scenario)
    debut = time.monotonic()
    resultat = _agent(s).run("Dis salut")
    assert time.monotonic() - debut >= 0.35
    assert resultat.usage.input_tokens == 10


def test_sans_usageMetadata_la_consommation_n_est_pas_inventee(serveur):
    from autoagent.schema import LLMRequest, Message

    def scenario(conn, vu):
        _entree_client(conn)
        _parle(conn, "Salut.")
    s = serveur(scenario)
    reponse = _agent(s).provider.complete(LLMRequest(messages=[
        Message(role="system", content="Tu es un robot."), Message(role="user", content="Dis salut")]))
    assert reponse.content == "Salut."
    assert reponse.usage is None                                        # jamais un zéro inventé


def test_le_json_du_serveur_est_compris_en_trame_texte_comme_en_trame_binaire(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        conn.texte(json.dumps({"serverContent": {"outputTranscription": {"text": "en texte"}}}))
        conn.texte(json.dumps({"serverContent": {"turnComplete": True}}))
    s = serveur(scenario)
    assert _agent(s).run("Dis quelque chose").output == "en texte"


# ── l'écoute : le run attend que l'utilisateur parle ───────────────────────────────────
def test_le_signal_d_ecoute_n_envoie_aucun_texte_et_attend_la_reponse_du_modele(serveur):
    def scenario(conn, vu):
        conn.sock.settimeout(0.5)
        try:
            vu["entrees"].append(_entree_client(conn))                  # ne doit RIEN recevoir
        except TimeoutError:
            pass
        time.sleep(0.2)                                                 # l'utilisateur finit de parler…
        _parle(conn, "Je t'écoute.")
    s = serveur(scenario)
    resultat = _agent(s).run(GeminiLiveProvider.LISTEN)
    assert resultat.output == "Je t'écoute."
    assert s.vu["entrees"] == []


def test_l_ecoute_attend_l_utilisateur_plus_longtemps_que_le_delai_de_reponse(serveur):
    """Quelqu'un peut se taire plusieurs minutes : en écoute, le délai `timeout` ne s'applique PAS (il borne le
    silence du serveur une fois qu'il a commencé à répondre). Trouvé par mutation : sans ce test, une écoute qui
    tombait en erreur après `timeout` secondes de silence passait toute la série."""
    def scenario(conn, vu):
        time.sleep(1.6)                                                 # l'utilisateur se tait plus longtemps que timeout
        _parle(conn, "Tu es là ?")
    s = serveur(scenario)
    config = ModelConfig(provider="gemini_live", model="gemini-3.8-live", api_key=CLE, base_url=s.url("/live"),
                         timeout=1.0)
    assert Agent.from_model_config(config).run(GeminiLiveProvider.LISTEN).output == "Tu es là ?"


def test_une_fois_la_reponse_commencee_un_serveur_muet_est_une_erreur_typee(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _json(conn, {"serverContent": {"outputTranscription": {"text": "Je comm"}}})
        time.sleep(2.5)                                                 # puis plus rien, sans fermer
    s = serveur(scenario)
    config = ModelConfig(provider="gemini_live", model="gemini-3.8-live", api_key=CLE, base_url=s.url("/live"),
                         timeout=1.0)
    with pytest.raises(ProviderError) as e:
        Agent.from_model_config(config).run("Raconte")
    assert e.value.retryable is True


def test_le_micro_envoie_de_l_audio_pendant_qu_un_run_attend_et_close_reveille_l_attente(serveur):
    recus: list[dict] = []

    def scenario(conn, vu):
        conn.sock.settimeout(8)
        for _ in range(3):
            recus.append(_entree_client(conn))
    s = serveur(scenario)
    agent = _agent(s)
    provider = agent.provider
    sortie: dict = {}

    def lecteur():
        try:
            agent.run(GeminiLiveProvider.LISTEN)
        except ProviderError as e:
            sortie["erreur"] = e
    fil = threading.Thread(target=lecteur)
    fil.start()
    assert provider.wait_ready(5)
    pcm = bytes(range(64)) * 5
    provider.send_audio(pcm)                                            # depuis un AUTRE fil que le run bloqué
    provider.send_audio(pcm[:100])
    provider.end_audio()
    time.sleep(0.4)
    provider.close()
    fil.join(5)
    assert not fil.is_alive(), "close() n'a pas réveillé le run qui attendait l'utilisateur"
    assert "erreur" in sortie
    assert recus[0] == {"realtimeInput": {"audio": {"data": base64.b64encode(pcm).decode(),
                                                    "mimeType": "audio/pcm;rate=16000"}}}
    assert recus[1]["realtimeInput"]["audio"]["data"] == base64.b64encode(pcm[:100]).decode()
    assert recus[2] == {"realtimeInput": {"audioStreamEnd": True}}


def test_open_connecte_avant_le_premier_run_et_send_image_envoie_une_image(serveur):
    recus: list[dict] = []

    def scenario(conn, vu):
        recus.append(_entree_client(conn))
    s = serveur(scenario)
    provider = _agent(s).provider
    provider.open("Tu es un robot.", [])
    assert provider.is_open
    provider.send_image(b"\xff\xd8JPEG\xff\xd9")
    time.sleep(0.3)
    provider.close()
    assert recus == [{"realtimeInput": {"video": {"data": base64.b64encode(b"\xff\xd8JPEG\xff\xd9").decode(),
                                                  "mimeType": "image/jpeg"}}}]


def test_send_text_glisse_une_phrase_dans_la_session_en_cours(serveur):
    """Pour l'hôte qui doit faire réagir le modèle à un événement (un réflexe : « tu vois une pomme ») pendant qu'un run
    attend l'utilisateur : la phrase part en `realtimeInput.text`, comme un texte de run."""
    recus: list[dict] = []

    def scenario(conn, vu):
        recus.append(_entree_client(conn))
    s = serveur(scenario)
    provider = _agent(s).provider
    provider.open("Tu es un robot.", [])
    provider.send_text("Tu vois une pomme.")
    time.sleep(0.3)
    provider.close()
    assert recus == [{"realtimeInput": {"text": "Tu vois une pomme."}}]


def test_send_text_vide_ou_sans_session_est_une_erreur_claire(serveur):
    provider = _agent(serveur(lambda conn, vu: None)).provider
    with pytest.raises(ProviderError, match="session"):
        provider.send_text("bonjour")
    with pytest.raises(ValueError, match="vide"):
        provider.send_text("   ")


def test_envoyer_de_l_audio_sans_session_est_une_erreur_claire(serveur):
    provider = _agent(serveur(lambda conn, vu: None)).provider
    with pytest.raises(ProviderError, match="session"):
        provider.send_audio(b"\x00\x01")


# ── les outils : le cœur de la série ───────────────────────────────────────────────────
def _agent_avec_outil(s, appels: list, **options):
    agent = _agent(s, **options)

    @agent.tool
    def avancer(cm: float) -> dict:
        """Fait avancer le robot de `cm` centimètres."""
        appels.append(cm)
        return {"ok": True, "parcouru_cm": cm}
    return agent


def test_un_appel_d_outil_est_execute_puis_la_reponse_finale_n_est_pas_vide(serveur):
    """LE défaut à attraper : le turnComplete périmé qui suit le toolCall ne doit PAS terminer le run."""
    def scenario(conn, vu):
        vu["entrees"].append(_entree_client(conn))
        _appel_outil(conn)
        vu["entrees"].append(_entree_client(conn))                       # la toolResponse
        _parle(conn, "J'avance de dix centimètres.")
    s = serveur(scenario)
    appels: list = []
    resultat = _agent_avec_outil(s, appels).run("Avance de dix centimètres")
    assert appels == [10]                                                # exécuté une fois, avec les bons arguments
    assert resultat.output == "J'avance de dix centimètres."
    assert resultat.steps == 2
    reponse = s.vu["entrees"][1]["toolResponse"]["functionResponses"][0]
    assert reponse["id"] == "call_42" and reponse["name"] == "avancer"   # rattachée à SON appel par l'identifiant
    assert "parcouru_cm" in json.dumps(reponse["response"])


def test_la_paire_perimee_qui_arrive_en_retard_est_aussi_absorbee(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _appel_outil(conn, paire_perimee=False)
        time.sleep(0.15)                                                 # elle arrive APRÈS le toolCall, pas dans le même paquet
        _json(conn, {"serverContent": {"generationComplete": True}})
        _json(conn, {"serverContent": {"turnComplete": True}})
        vu["entrees"].append(_entree_client(conn))
        _parle(conn, "Fait.")
    s = serveur(scenario)
    assert _agent_avec_outil(s, []).run("Avance").output == "Fait."


def test_sans_paire_perimee_le_run_continue_apres_un_court_delai_de_grace(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _appel_outil(conn, paire_perimee=False)
        vu["entrees"].append(_entree_client(conn))
        _parle(conn, "Fait.")
    s = serveur(scenario)
    agent = _agent_avec_outil(s, [])
    agent.provider.GRACE_S = 0.2
    debut = time.monotonic()
    assert agent.run("Avance").output == "Fait."
    assert time.monotonic() - debut < 4.0


def test_deux_appels_en_parallele_recoivent_chacun_leur_reponse_dans_un_seul_message(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _json(conn, {"toolCall": {"functionCalls": [
            {"name": "avancer", "args": {"cm": 5}, "id": "a"}, {"name": "avancer", "args": {"cm": 7}, "id": "b"}]}})
        _json(conn, {"serverContent": {"generationComplete": True}})
        _json(conn, {"serverContent": {"turnComplete": True}})
        vu["entrees"].append(_entree_client(conn))
        _parle(conn, "Fait.")
    s = serveur(scenario)
    appels: list = []
    _agent_avec_outil(s, appels).run("Avance deux fois")
    reponses = s.vu["entrees"][0]["toolResponse"]["functionResponses"]
    assert sorted(r["id"] for r in reponses) == ["a", "b"]
    assert sorted(appels) == [5, 7]


def test_une_politique_qui_refuse_empeche_l_outil_de_tourner_et_gemini_recoit_le_refus(serveur):
    """La porte est la MÊME que pour les autres fournisseurs : la boucle ne change pas, donc le refus s'applique."""
    def scenario(conn, vu):
        _entree_client(conn)
        _appel_outil(conn)
        vu["entrees"].append(_entree_client(conn))
        _parle(conn, "Je n'ai pas le droit.")
    s = serveur(scenario)
    appels: list = []
    agent = _agent_avec_outil(
        s, appels, tool_policy=lambda ctx: "interdit par la politique du robot" if ctx.call.name == "avancer" else None)
    resultat = agent.run("Avance de dix centimètres")
    assert appels == [], "l'outil a tourné malgré le refus de la politique"
    assert "interdit par la politique du robot" in json.dumps(s.vu["entrees"][0]["toolResponse"])
    assert resultat.output == "Je n'ai pas le droit."


def test_un_outil_qui_plante_devient_une_erreur_que_gemini_voit_jamais_un_plantage(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _appel_outil(conn)
        vu["entrees"].append(_entree_client(conn))
        _parle(conn, "Ça a échoué.")
    s = serveur(scenario)
    agent = _agent(s)

    @agent.tool
    def avancer(cm: float) -> dict:
        """Fait avancer le robot."""
        raise RuntimeError("moteur bloqué")
    assert agent.run("Avance").output == "Ça a échoué."
    assert "moteur bloqué" in json.dumps(s.vu["entrees"][0]["toolResponse"], ensure_ascii=False)


def test_la_garde_anti_boucle_s_applique_aussi_a_la_voix(serveur):
    """Un modèle qui rappelle indéfiniment le même outil est coupé par la boucle, comme avec n'importe quel fournisseur."""
    def scenario(conn, vu):
        _entree_client(conn)
        for i in range(6):
            _appel_outil(conn, identifiant=f"c{i}")
            vu["entrees"].append(_entree_client(conn))
        _parle(conn, "J'arrête.")
    s = serveur(scenario)
    appels: list = []
    agent = _agent_avec_outil(s, appels, max_repeated_tool_calls=2, max_steps=8)
    agent.run("Avance")
    assert len(appels) < 6, "la garde anti-boucle n'a pas coupé les appels répétés"


# ── barge-in, coupures, reprise ──────────────────────────────────────────────────────
def test_l_interruption_vide_le_haut_parleur_et_termine_le_tour_avec_ce_qui_a_ete_dit(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _json(conn, _audio(b"\x01\x02" * 30))
        _json(conn, {"serverContent": {"outputTranscription": {"text": "Je commence à par"}}})
        _json(conn, {"serverContent": {"interrupted": True}})
        _json(conn, {"serverContent": {"turnComplete": True}})
    s = serveur(scenario)
    agent = _agent(s)
    coupures: list = []
    agent.provider.on_interrupted = lambda: coupures.append(True)
    resultat = agent.run("Raconte une histoire")
    assert coupures == [True]
    assert resultat.output == "Je commence à par"


def test_une_interruption_sans_turnComplete_ne_bloque_pas_le_run(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _json(conn, {"serverContent": {"outputTranscription": {"text": "Début"}}})
        _json(conn, {"serverContent": {"interrupted": True}})
    s = serveur(scenario)
    agent = _agent(s)
    agent.provider.GRACE_S = 0.2
    assert agent.run("Raconte").output == "Début"


@pytest.mark.parametrize("code,retryable", [(1008, False), (1007, False), (1011, True), (1001, True), (1013, True)])
def test_une_fermeture_du_serveur_est_typee_et_le_code_dit_si_on_peut_reessayer(serveur, code, retryable):
    def scenario(conn, vu):
        _entree_client(conn)
        conn.fermer_proprement(code, "raison du serveur")
    s = serveur(scenario)
    with pytest.raises(ProviderError) as e:
        _agent(s).run("Dis bonjour")
    assert str(code) in str(e.value) and "raison du serveur" in str(e.value)
    assert e.value.retryable is retryable


def test_une_coupure_brutale_en_cours_de_reponse_n_est_jamais_rendue_comme_un_succes(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _json(conn, {"serverContent": {"outputTranscription": {"text": "Je commence une phr"}}})
        # puis la connexion tombe, sans turnComplete ni trame de fermeture
    s = serveur(scenario)
    with pytest.raises(ProviderError) as e:
        _agent(s).run("Raconte")
    assert e.value.retryable is True


def test_apres_une_coupure_le_run_suivant_se_reconnecte_avec_le_jeton_de_reprise(serveur):
    def scenario(conn, vu):
        if len(vu["setups"]) == 1:
            _entree_client(conn)
            _json(conn, {"sessionResumptionUpdate": {"newHandle": "reprise-1", "resumable": True}})
            _parle(conn, "Premier tour.")
            _json(conn, {"goAway": {"timeLeft": "30s"}})
            conn.fermer_proprement(1000, "fin de session")
        else:
            _entree_client(conn)
            _parle(conn, "Second tour.")
    s = serveur(scenario)
    agent = _agent(s)
    avertissements: list = []
    agent.provider.on_go_away = avertissements.append
    assert agent.run("Un").output == "Premier tour."
    time.sleep(0.3)                                                      # la fermeture du serveur arrive
    assert agent.run("Deux").output == "Second tour."
    assert len(s.vu["setups"]) == 2
    assert s.vu["setups"][1]["setup"]["sessionResumption"] == {"handle": "reprise-1"}
    assert avertissements == ["30s"]


def test_un_jeton_de_reprise_refuse_donne_une_nouvelle_session_sans_lui(serveur):
    """Un handle expiré ne doit pas bloquer la voix : le serveur le refuse, on repart d'une session neuve."""
    def scenario(conn, vu):
        if len(vu["setups"]) == 1:
            _entree_client(conn)
            _json(conn, {"sessionResumptionUpdate": {"newHandle": "perime", "resumable": True}})
            _parle(conn, "Premier tour.")
            conn.fermer_proprement(1000, "fin")
        else:
            _entree_client(conn)
            _parle(conn, "Nouveau départ.")
    s = serveur(scenario)
    s.vu["refus"] = lambda setup: ("handle expiré"
                                   if setup["setup"].get("sessionResumption", {}).get("handle") == "perime" else None)
    agent = _agent(s)
    assert agent.run("Un").output == "Premier tour."
    time.sleep(0.3)
    assert agent.run("Deux").output == "Nouveau départ."
    assert len(s.vu["setups"]) == 3                                     # 1 initiale, 1 refusée (avec handle), 1 neuve
    assert s.vu["setups"][1]["setup"]["sessionResumption"] == {"handle": "perime"}
    assert "sessionResumption" not in s.vu["setups"][2]["setup"]


def test_une_erreur_dans_un_crochet_ne_tue_pas_la_lecture(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _parle(conn, "Salut.", (b"\x01\x02", b"\x03\x04"))
    s = serveur(scenario)
    agent = _agent(s)

    def casse(_pcm):
        raise RuntimeError("le haut-parleur plante")
    agent.provider.on_audio = casse
    assert agent.run("Dis salut").output == "Salut."


def test_un_changement_d_outils_ouvre_une_nouvelle_session_avec_les_nouvelles_declarations(serveur):
    def scenario(conn, vu):
        _entree_client(conn)
        _parle(conn, "Un.")
    s = serveur(scenario)
    agent = _agent(s)
    agent.run("Un")
    time.sleep(0.2)

    @agent.tool
    def tourner(degres: float) -> dict:
        """Fait tourner le robot."""
        return {"ok": True}
    agent.run("Deux")
    assert len(s.vu["setups"]) == 2
    assert s.vu["setups"][1]["setup"]["tools"][0]["functionDeclarations"][0]["name"] == "tourner"
