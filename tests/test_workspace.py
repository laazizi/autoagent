import tempfile
from pathlib import Path

import pytest

from autoagent.workspace import ProjectWorkspace, WorkspaceError


@pytest.fixture
def tmp_workspace() -> ProjectWorkspace:
    tmp = Path(tempfile.mkdtemp(prefix="autoagent_test_"))
    return ProjectWorkspace(root=tmp)


class TestFileReadWrite:
    def test_write_and_read(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("hello.txt", "Hello World", reason="test")
        result = tmp_workspace.read_file("hello.txt")
        assert result["content"] == "Hello World"
        assert result["chars"] == 11
        assert result["truncated"] is False

    def test_read_nonexistent_file_raises(self, tmp_workspace: ProjectWorkspace) -> None:
        with pytest.raises(WorkspaceError, match="does not exist"):
            tmp_workspace.read_file("nope.txt")

    def test_write_changes_recorded(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("a.txt", "AAA", reason="create a")
        changes = tmp_workspace.list_changes()
        assert len(changes["changes"]) == 1
        assert changes["changes"][0]["created"] is True

    def test_list_files(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("src/a.py", "pass", reason="a")
        tmp_workspace.write_file("src/b.py", "pass", reason="b")
        tmp_workspace.write_file("README.md", "# Hello", reason="readme")
        result = tmp_workspace.list_files(pattern="src/*.py")
        assert set(result["files"]) == {"src/a.py", "src/b.py"}

    def test_list_files_max_limit(self, tmp_workspace: ProjectWorkspace) -> None:
        for i in range(5):
            tmp_workspace.write_file(f"f{i}.txt", f"content {i}", reason=f"file {i}")
        result = tmp_workspace.list_files(max_files=3)
        assert len(result["files"]) <= 3


class TestReplaceText:
    def test_simple_replace(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("config.py", "DEBUG = False", reason="init")
        result = tmp_workspace.replace_text("config.py", "False", "True", reason="enable debug")
        assert result["replaced"] == 1
        content = tmp_workspace.read_file("config.py")["content"]
        assert content == "DEBUG = True"

    def test_replace_count(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("data.txt", "aa aa aa", reason="init")
        result = tmp_workspace.replace_text("data.txt", "aa", "bb", count=2)
        assert result["replaced"] == 2
        content = tmp_workspace.read_file("data.txt")["content"]
        assert content == "bb bb aa"

    def test_replace_nonexistent_text_raises(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("data.txt", "hello", reason="init")
        with pytest.raises(WorkspaceError, match="not found"):
            tmp_workspace.replace_text("data.txt", "zzz", "yyy")

    def test_replace_empty_old_raises(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("data.txt", "hello", reason="init")
        with pytest.raises(WorkspaceError, match="cannot be empty"):
            tmp_workspace.replace_text("data.txt", "", "yyy")


class TestRollback:
    def test_rollback_last_write(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("f.txt", "version-1", reason="v1")
        tmp_workspace.write_file("f.txt", "version-2", reason="v2")
        assert tmp_workspace.read_file("f.txt")["content"] == "version-2"
        tmp_workspace.rollback_last_change()
        assert tmp_workspace.read_file("f.txt")["content"] == "version-1"

    def test_rollback_undoes_file_creation(self, tmp_workspace: ProjectWorkspace) -> None:
        # Write creates a file (before=None, after=content) → rollback deletes it
        tmp_workspace.write_file("removed.txt", "temporary")
        tmp_workspace.rollback_last_change()
        assert not tmp_workspace.resolve("removed.txt").exists()

    def test_rollback_undoes_modification(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("data.txt", "original")
        # Second write replaces the file
        tmp_workspace.write_file("data.txt", "modified")
        tmp_workspace.rollback_last_change()
        assert tmp_workspace.read_file("data.txt")["content"] == "original"

    def test_rollback_by_id(self, tmp_workspace: ProjectWorkspace) -> None:
        tmp_workspace.write_file("a.txt", "A1", reason="a1")
        r2 = tmp_workspace.write_file("b.txt", "B1", reason="b1")
        cid = r2["change"]["id"]
        tmp_workspace.write_file("a.txt", "A2", reason="a2")
        tmp_workspace.rollback_change(cid)
        assert tmp_workspace.read_file("a.txt")["content"] == "A1"
        assert not tmp_workspace.resolve("b.txt").exists()

    def test_rollback_empty_raises(self, tmp_workspace: ProjectWorkspace) -> None:
        with pytest.raises(WorkspaceError, match="No changes"):
            tmp_workspace.rollback_last_change()


class TestPathValidation:
    def test_absolute_path_blocked(self, tmp_workspace: ProjectWorkspace) -> None:
        from pathlib import Path

        # Use an absolute path that works on both Windows and POSIX.
        # On Windows a rooted path like C:\... is absolute;
        # on POSIX /etc/passwd is absolute.  We use pathlib to pick one.
        abs_path = str(Path(tmp_workspace.root.anchor) / "outside.txt")
        with pytest.raises(WorkspaceError, match="Absolute paths"):
            tmp_workspace.write_file(abs_path, "bad")

    def test_path_escape_blocked(self, tmp_workspace: ProjectWorkspace) -> None:
        with pytest.raises(WorkspaceError, match="escapes"):
            tmp_workspace.write_file("../outside.txt", "bad")

    def test_ignored_dir_blocked(self, tmp_workspace: ProjectWorkspace) -> None:
        git_dir = tmp_workspace.root / ".git"
        git_dir.mkdir()
        with pytest.raises(WorkspaceError, match="ignored"):
            tmp_workspace.read_file(".git/config")


class TestWriteValidation:
    def test_extension_filtering(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="autoagent_test_"))
        ws = ProjectWorkspace(root=tmp, allowed_write_extensions={".py", ".txt"})
        ws.write_file("ok.py", "pass", reason="test")
        with pytest.raises(WorkspaceError, match="blocked"):
            ws.write_file("bad.exe", "virus", reason="test")

    def test_max_write_chars(self, tmp_workspace: ProjectWorkspace) -> None:
        ws = ProjectWorkspace(root=tmp_workspace.root, max_write_chars=10)
        with pytest.raises(WorkspaceError, match="exceeds"):
            ws.write_file("big.txt", "x" * 100)


class TestReadTruncation:
    def test_truncation(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="autoagent_test_"))
        ws = ProjectWorkspace(root=tmp, max_read_chars=5)
        ws.write_file("long.txt", "1234567890", reason="test")
        result = ws.read_file("long.txt")
        assert result["truncated"] is True
        assert len(result["content"]) <= 5


# ---------------------------------------------------------------------------
# COMPOSABILITÉ TEMPORELLE — la propriété globale, pas les cas un par un.
#
# Formulation empruntée à « A Programming Paradigm for Spatiotemporal
# Composability » (Shi, Zhang, Cui — PKU / DeepSeek-AI, 2026), qui formalise
# l'effet réversible : « every context transformation carries an inverse that
# the runtime tracks ». `ProjectWorkspace` implémente exactement ça avec son
# change history. Les tests existants vérifient des annulations INDIVIDUELLES ;
# ceux-ci vérifient le théorème : annuler TOUT ramène à l'état initial, à
# l'octet près — et par deux chemins différents (en bloc, ou un par un), ce qui
# est la saveur « confluence » de leur métathéorie.
# ---------------------------------------------------------------------------


def _empreinte(racine: Path) -> dict[str, str]:
    """L'état observable de l'arbre : chemin relatif → contenu."""
    return {
        str(p.relative_to(racine)).replace("\\", "/"): p.read_text(encoding="utf-8")
        for p in sorted(racine.rglob("*"))
        if p.is_file()
    }


def _sequence(ws: ProjectWorkspace) -> None:
    """Une suite d'écritures représentative : créations, écrasements, sous-dossiers."""
    ws.write_file("a.txt", "a1", reason="creation")
    ws.write_file("dossier/b.txt", "b1", reason="creation en sous-dossier")
    ws.write_file("a.txt", "a2", reason="ecrasement")
    ws.write_file("c.txt", "c1", reason="creation")
    ws.write_file("dossier/b.txt", "b2", reason="ecrasement")
    ws.write_file("a.txt", "a3", reason="ecrasement")


class TestComposabiliteTemporelle:
    def test_tout_annuler_en_bloc_restaure_l_etat_initial(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="autoagent_test_"))
        ws = ProjectWorkspace(root=tmp)
        ws.write_file("prealable.txt", "je préexiste", reason="etat initial")
        ws._changes.clear()                     # cet écrit fait partie de l'état de départ
        avant = _empreinte(tmp)

        _sequence(ws)
        assert _empreinte(tmp) != avant, "la séquence n'a rien modifié : test vide"

        premier = ws.list_changes()["changes"][0]["id"]
        ws.rollback_change(premier)             # annule celui-ci ET tous les suivants

        assert _empreinte(tmp) == avant, "l'annulation en bloc ne restaure pas l'état initial"
        assert ws.list_changes()["changes"] == [], "le journal n'est pas vidé"

    def test_annuler_un_par_un_donne_le_meme_etat(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="autoagent_test_"))
        ws = ProjectWorkspace(root=tmp)
        ws.write_file("prealable.txt", "je préexiste", reason="etat initial")
        ws._changes.clear()
        avant = _empreinte(tmp)

        _sequence(ws)
        for _ in range(len(ws.list_changes()["changes"])):
            ws.rollback_last_change()

        assert _empreinte(tmp) == avant, "l'annulation pas à pas divergeait de l'état initial"
        assert ws.list_changes()["changes"] == []

    def test_un_fichier_cree_puis_annule_ne_laisse_pas_de_dossier_fantome(self) -> None:
        """Le contenu revient, mais le dossier créé au passage, lui, reste.

        Ce n'est PAS un bug : c'est la limite exacte de la réversibilité ici, et
        elle mérite d'être écrite noir sur blanc plutôt que découverte un jour.
        """
        tmp = Path(tempfile.mkdtemp(prefix="autoagent_test_"))
        ws = ProjectWorkspace(root=tmp)
        ws.write_file("neuf/fichier.txt", "x", reason="creation")
        ws.rollback_last_change()

        assert not (tmp / "neuf" / "fichier.txt").exists(), "le fichier n'a pas été retiré"
        assert (tmp / "neuf").is_dir(), (
            "si ce dossier disparaît un jour, tant mieux — mets à jour ce test")
