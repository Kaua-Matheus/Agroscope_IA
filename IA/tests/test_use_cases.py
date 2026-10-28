"""
Casos de uso da API (rodar dentro de Main/Project/IA):

    python -m unittest discover -s tests -v

Cada teste simula o que um usuário do aplicativo faria e verifica a resposta da API.
Os modelos (models/...) precisam existir; sem eles os testes são pulados.
"""
import base64
import io
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Modelos usados nos testes (sobrescrevem o .env, pois load_dotenv não substitui variáveis já definidas)
os.environ.update({
    "COFFEE": "./models/Coffee/coffee.pth",
    "SOYBEAN": "./models/Soybean/soybean_v2.pth",
    "WHEAT": "./models/Wheat/wheat.pth",
    "GATE": "./models/Gate/gate.pth",
})
os.environ["TOMATO"] = ""  # vazio = cultura desligada (load_dotenv não sobrescreve variáveis existentes)
os.environ["CORN"] = ""

MODELS_OK = all(os.path.isfile(os.environ[k]) for k in ("COFFEE", "SOYBEAN", "WHEAT", "GATE"))

from fastapi.testclient import TestClient  # noqa: E402
import Main.IA.app as app_module  # noqa: E402


def jpeg_bytes(img :Image.Image, quality :int=90, exif=None) -> bytes:
    buf = io.BytesIO()
    if exif is not None:
        img.save(buf, "JPEG", quality=quality, exif=exif)
    else:
        img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def fixture(name :str) -> Image.Image:
    return Image.open(os.path.join(FIX, name)).convert("RGB")


def fake_document() -> Image.Image:
    """Foto de um papel com texto (ex.: usuário enviou a foto errada)."""
    img = Image.new("RGB", (900, 1200), (245, 245, 240))
    d = ImageDraw.Draw(img)
    for i in range(40):
        y = 60 + i * 27
        d.rectangle([60, y, 60 + (i * 97) % 700 + 100, y + 10], fill=(40, 40, 40))
    return img


def fake_sky() -> Image.Image:
    """Céu azul (foto tirada sem querer para cima)."""
    a = np.zeros((900, 1200, 3), np.uint8)
    a[..., 0] = np.linspace(90, 170, 900)[:, None]
    a[..., 1] = np.linspace(140, 200, 900)[:, None]
    a[..., 2] = 235
    return Image.fromarray(a)


def fake_wall() -> Image.Image:
    """Parede de tijolos."""
    img = Image.new("RGB", (1200, 900), (150, 60, 40))
    d = ImageDraw.Draw(img)
    for row in range(0, 900, 60):
        off = 60 if (row // 60) % 2 else 0
        d.line([(0, row), (1200, row)], fill=(200, 200, 190), width=6)
        for x in range(off, 1200, 120):
            d.line([(x, row), (x, row + 60)], fill=(200, 200, 190), width=6)
    return img.filter(ImageFilter.GaussianBlur(1))


@unittest.skipUnless(MODELS_OK, "modelos não encontrados em ./models")
class UseCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = TestClient(app_module.APP)
        cls.client = cls.ctx.__enter__()  # dispara o lifespan (carrega os modelos uma vez)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def predict(self, culture, data, filename="foto.jpg", content_type="image/jpeg", **form):
        fields = {"culture": culture, **{k: str(v).lower() for k, v in form.items()}}
        return self.client.post("/predict", data=fields, files={"file": (filename, data, content_type)})

    # ---------- informações ----------
    def test_01_health(self):
        """A API sobe e informa as culturas carregadas e se a triagem por modelo está ativa."""
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(set(body["cultures"]), {"coffee", "soybean", "wheat"})
        self.assertTrue(body["gate"])

    def test_02_cultures_mostra_tipo_de_modelo(self):
        """Modelos novos usam recorte da folha; modelos antigos mantêm o preparo com que foram treinados."""
        r = self.client.get("/cultures").json()
        by = {c["culture"]: c for c in r["cultures"]}
        self.assertIn("LeafCrop", by["coffee"]["preprocessing"])
        self.assertIn("legacy", by["wheat"]["preprocessing"])
        self.assertIn("tomato", r["unavailable"])

    def test_03_modelinfo_compatibilidade(self):
        """Rota antiga /modelinfo continua respondendo no formato anterior."""
        r = self.client.get("/modelinfo").json()
        self.assertEqual(r["models"]["coffee"]["num_classes"], 3)

    # ---------- diagnóstico ----------
    def test_10_cafe_saudavel(self):
        """Produtor envia foto de folha de café saudável."""
        r = self.predict("coffee", jpeg_bytes(fixture("coffee_healthy.jpg")))
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["diagnosis"]["predicted_class"], "Healthy")
        self.assertTrue(body["diagnosis"]["is_healthy"])
        self.assertEqual(body["gate"]["verdict"], "ok")
        self.assertIsNotNone(body["diagnosis"]["leaf_bbox"])  # folha foi recortada antes do modelo

    def test_11_cafe_phoma(self):
        r = self.predict("coffee", jpeg_bytes(fixture("coffee_phoma.jpg")))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["diagnosis"]["predicted_class"], "Phoma")

    def test_12_cafe_ferrugem(self):
        r = self.predict("coffee", jpeg_bytes(fixture("coffee_rust.jpg")))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["diagnosis"]["predicted_class"], "Rust")

    def test_13_soja_com_apelido_em_portugues(self):
        """Front pode mandar 'soja' / 'Café' — apelidos e acentos são aceitos."""
        r = self.predict("Soja", jpeg_bytes(fixture("soy_rust.jpg")))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["culture"], "soybean")
        self.assertEqual(r.json()["diagnosis"]["predicted_class"], "Rust")
        r = self.predict("Café", jpeg_bytes(fixture("coffee_rust.jpg")))
        self.assertEqual(r.json()["culture"], "coffee")

    def test_14_soja_crestamento(self):
        r = self.predict("soybean", jpeg_bytes(fixture("soy_blight.jpg")))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["diagnosis"]["predicted_class"], "Bacterial_Blight")

    def test_15_campos_antigos_para_o_front(self):
        """O front atual lê expert.predict (maiúsculo) e expert.predict_confidence (0-100)."""
        body = self.predict("coffee", jpeg_bytes(fixture("coffee_rust.jpg"))).json()
        self.assertEqual(body["expert"]["predict"], body["diagnosis"]["predicted_class"].upper())
        self.assertAlmostEqual(body["expert"]["predict_confidence"], body["diagnosis"]["confidence"] * 100, delta=0.01)
        self.assertAlmostEqual(sum(body["diagnosis"]["probabilities"].values()), 1.0, delta=0.01)

    def test_16_modelo_antigo_trigo(self):
        """Cultura com modelo antigo (v1) continua funcionando, com o preparo antigo."""
        r = self.predict("trigo", jpeg_bytes(fixture("wheat_leaf.jpg")))
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()["diagnosis"]
        self.assertTrue(d["model"]["legacy_preprocessing"])
        self.assertIsNone(d["leaf_bbox"])

    def test_17_mapa_de_calor(self):
        """Com explain=true a resposta traz o Grad-CAM como PNG em base64."""
        r = self.predict("coffee", jpeg_bytes(fixture("coffee_phoma.jpg")), explain=True)
        exp = r.json()["diagnosis"]["explanation"]
        png = Image.open(io.BytesIO(base64.b64decode(exp["heatmap_png_base64"])))
        self.assertEqual(png.format, "PNG")
        self.assertTrue(0 <= exp["peak_xy"][0] <= 1 and 0 <= exp["peak_xy"][1] <= 1)

    def test_18_foto_de_celular_rotacionada(self):
        """Celular grava a foto deitada + tag EXIF de rotação; resultado deve ser o mesmo."""
        img = fixture("coffee_rust.jpg")
        exif = Image.Exif()
        exif[274] = 6  # "girar 90° para exibir"
        rotated = img.rotate(90, expand=True)
        r = self.predict("coffee", jpeg_bytes(rotated, exif=exif.tobytes()))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["diagnosis"]["predicted_class"], "Rust")

    def test_19_png_com_transparencia(self):
        """PNG com canal alfa (ex.: imagem recortada em outro app)."""
        img = fixture("coffee_healthy.jpg").convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        r = self.predict("coffee", buf.getvalue(), filename="folha.png", content_type="image/png")
        self.assertEqual(r.status_code, 200, r.text)

    # ---------- triagem: imagem que não é folha ----------
    def test_20_nao_folha_documento(self):
        r = self.predict("coffee", jpeg_bytes(fake_document()))
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["reason"], "not_leaf")

    def test_21_nao_folha_ceu(self):
        r = self.predict("coffee", jpeg_bytes(fake_sky()))
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["reason"], "not_leaf")

    def test_22_nao_folha_parede(self):
        r = self.predict("coffee", jpeg_bytes(fake_wall()))
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["reason"], "not_leaf")

    def test_23_nao_folha_nao_passa_nem_com_force(self):
        """force=true só ignora qualidade; nunca libera imagem que não é folha."""
        r = self.predict("coffee", jpeg_bytes(fake_document()), force=True)
        self.assertEqual(r.status_code, 422)

    # ---------- triagem: qualidade ----------
    def test_30_foto_tremida(self):
        """Foto muito desfocada é recusada com mensagem orientando nova foto."""
        img = fixture("coffee_rust.jpg").filter(ImageFilter.GaussianBlur(14))
        r = self.predict("coffee", jpeg_bytes(img))
        self.assertEqual(r.status_code, 422, r.text)
        body = r.json()
        self.assertEqual(body["reason"], "low_quality")
        self.assertIn("foto", body["message"])

    def test_31_foto_tremida_com_force(self):
        """Usuário insiste (force=true): analisa, mas marca o resultado para revisão."""
        img = fixture("coffee_rust.jpg").filter(ImageFilter.GaussianBlur(14))
        r = self.predict("coffee", jpeg_bytes(img), force=True)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["diagnosis"]["status"], "review")
        self.assertIn("qualidade da imagem", r.json()["diagnosis"]["review_reasons"])

    def test_32_foto_escura_passa_com_aviso(self):
        """Foto escura NÃO derruba o acerto dos especialistas (medido: -0 a -2 pts), então passa — só com aviso."""
        a = (np.asarray(fixture("coffee_rust.jpg")).astype(np.float32) * 0.2).astype(np.uint8)
        r = self.predict("coffee", jpeg_bytes(Image.fromarray(a)))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("imagem escura", r.json()["gate"]["warnings"])

    def test_35_foto_estourada(self):
        """Foto clara demais (estourada) derruba o acerto (-16 a -29 pts) e é recusada."""
        a = np.clip(np.asarray(fixture("coffee_rust.jpg")).astype(np.float32) * 2.6, 0, 255).astype(np.uint8)
        r = self.predict("coffee", jpeg_bytes(Image.fromarray(a)))
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["reason"], "low_quality")

    def test_36_compressao_extrema(self):
        """JPEG muito comprimido (ex.: imagem encaminhada várias vezes por mensageiro)."""
        r = self.predict("coffee", jpeg_bytes(fixture("coffee_rust.jpg"), quality=3))
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["reason"], "low_quality")

    def test_33_imagem_minuscula(self):
        img = fixture("coffee_rust.jpg").resize((64, 64))
        r = self.predict("coffee", jpeg_bytes(img))
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["reason"], "low_quality")

    def test_34_desfoque_leve_passa(self):
        """Desfoque leve não deve barrar o usuário (não prejudica o diagnóstico)."""
        img = fixture("coffee_rust.jpg").filter(ImageFilter.GaussianBlur(1.5))
        r = self.predict("coffee", jpeg_bytes(img, quality=60))
        self.assertEqual(r.status_code, 200, r.text)

    # ---------- erros de entrada ----------
    def test_40_cultura_desconhecida(self):
        r = self.predict("banana", jpeg_bytes(fixture("coffee_rust.jpg")))
        self.assertEqual(r.status_code, 400)
        self.assertIn("coffee", r.json()["detail"]["available"])

    def test_41_cultura_sem_modelo_carregado(self):
        r = self.predict("tomate", jpeg_bytes(fixture("coffee_rust.jpg")))
        self.assertEqual(r.status_code, 400)

    def test_42_arquivo_que_nao_e_imagem(self):
        r = self.predict("coffee", b"isto nao e uma imagem", filename="a.txt", content_type="text/plain")
        self.assertEqual(r.status_code, 415)

    def test_43_imagem_corrompida(self):
        data = jpeg_bytes(fixture("coffee_rust.jpg"))[:300]
        r = self.predict("coffee", data)
        self.assertEqual(r.status_code, 415, r.text)

    def test_44_arquivo_vazio(self):
        r = self.predict("coffee", b"")
        self.assertEqual(r.status_code, 400)

    def test_45_arquivo_grande_demais(self):
        old = app_module.MAX_UPLOAD_BYTES
        app_module.MAX_UPLOAD_BYTES = 50 * 1024
        try:
            r = self.predict("coffee", jpeg_bytes(fixture("coffee_rust.jpg")))
            self.assertEqual(r.status_code, 413)
        finally:
            app_module.MAX_UPLOAD_BYTES = old

    # ---------- carga ----------
    def test_50_varios_usuarios_ao_mesmo_tempo(self):
        """8 requisições simultâneas (com Grad-CAM) devem responder igual a uma requisição isolada."""
        data = jpeg_bytes(fixture("coffee_phoma.jpg"))
        alone = self.predict("coffee", data, explain=True).json()["diagnosis"]
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(lambda _: self.predict("coffee", data, explain=True), range(8)))
        for r in results:
            self.assertEqual(r.status_code, 200)
            d = r.json()["diagnosis"]
            self.assertEqual(d["predicted_class"], alone["predicted_class"])
            self.assertAlmostEqual(d["confidence"], alone["confidence"], places=3)
            self.assertEqual(d["explanation"]["peak_xy"], alone["explanation"]["peak_xy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
