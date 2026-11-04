"""
Executa os casos de teste manuais da IA (casos.json) contra a API em execução e grava resultados.json.

    python app.py                                                       # em outro terminal, dentro de packages/IA
    python tests/manual/executar_casos.py <pasta das imagens> [URL]     # URL padrão: http://localhost:5000

A pasta das imagens fica fora do Git, organizada em <Cultura>/<Classe>/ e _Outros/; os caminhos estão em casos.json.
O limite de 80% é o mesmo do PredictUseCase do backend: abaixo dele o cliente não vê o diagnóstico.
"""
import json
import mimetypes
import sys
from pathlib import Path

import httpx

AQUI = Path(__file__).resolve().parent
LIMITE_EXIBICAO = 0.8


def _pct(valor :float) -> str:
    return f"{valor:.1%}".replace(".", ",")


def avaliar(caso :dict, resp :httpx.Response) -> tuple[bool, str]:
    body = resp.json()
    if resp.status_code == 422 and body.get("status") == "rejected":
        texto = f"Recusada na triagem ({body['reason']}; folha: {_pct((body['gate'].get('scores') or {}).get('leaf', 0))}). Mensagem: {body['message']}"
        return caso["criterio"] != "diagnostico", texto
    if resp.status_code != 200:
        return False, f"Erro HTTP {resp.status_code}: {resp.text[:200]}"

    d = body["diagnosis"]
    exibido = d["confidence"] >= LIMITE_EXIBICAO
    texto = (f"Triagem aceitou (folha: {_pct(body['gate']['scores']['leaf'])}). Modelo: {d['predicted_class']} com {_pct(d['confidence'])}. "
             + ("Cliente vê o diagnóstico." if exibido else "Abaixo de 80%: backend pede nova foto."))
    if caso["criterio"] == "diagnostico":
        return exibido and d["predicted_class"] == caso["classe_esperada"], texto
    if caso["criterio"] == "sem_diagnostico_confiante":
        return not exibido, texto
    return False, texto  # recusa_triagem: a foto não deveria ter passado da triagem


def main(imagens :Path, url :str):
    casos = json.loads((AQUI / "casos.json").read_text())
    resultados = []
    with httpx.Client(base_url=url, timeout=120) as client:
        for caso in casos:
            foto = imagens / caso["imagem"]
            tipo = mimetypes.guess_type(foto.name)[0] or "image/jpeg"
            resp = client.post("/predict", data={"culture": caso["cultura"]}, files={"file": (foto.name, foto.read_bytes(), tipo)})
            aprovado, texto = avaliar(caso, resp)
            body = resp.json()
            resultados.append({
                "id": caso["id"], "status": "Aprovado" if aprovado else "Reprovado", "obtido": texto, "http": resp.status_code,
                "probabilidades": (body.get("diagnosis") or {}).get("probabilities"), "gate": body.get("gate"),
            })
            print(f"{caso['id']}  {resultados[-1]['status']:9}  {texto}")
    (AQUI / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=1))
    print(sum(r["status"] == "Aprovado" for r in resultados), "de", len(resultados), "aprovados")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000")
