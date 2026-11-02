"""Live search of real papers (arXiv + OpenAlex) - no API keys needed, only stdlib."""
import hashlib, os, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

UA = "ApexResearchAI/1.0 (hackathon project; mailto:%s)" % os.getenv("APEX_MAILTO", "none@example.com")
CACHE_TTL = 3600
_cache = {}

STOP = set("""about above after again also among because been before being between both build can could does done each
either from have having here however into just like made make many more most much must only other over same should since
some such than that their them then there these they this those through thus under until upon used uses using very want
were what when where which while will with within without would your paper proposal propose proposed project research
study work based approach method methods system systems results show shows shown present presented novel new main
different various several first second third one two three also well may might use our""".split())

def http_get(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

def keywords(text, title="", n=8):
    words = re.findall(r"[A-Za-z]{4,}", (title + " ") * 3 + text)
    c = Counter(w.lower() for w in words if w.lower() not in STOP)
    return [w for w, _ in sorted(c.items(), key=lambda x: (-x[1], -len(x[0])))[:n]]

def _clean(s):
    return re.sub(r"\s+", " ", s or "").strip()

def arxiv_fulltext(url, max_bytes=12000000):
    """Best effort full text extraction for an arXiv paper PDF; returns empty on failure."""
    import io
    try:
        ident = url.rstrip("/").rsplit("/", 1)[-1]
        if not re.match(r"\d{4}\.\d{4,5}(v\d+)?$", ident):
            return ""
        req = urllib.request.Request("https://export.arxiv.org/pdf/" + ident, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = r.read(max_bytes + 1)
        if len(data) > max_bytes or not data.startswith(b"%PDF"):
            return ""
        from pypdf import PdfReader
        pages = PdfReader(io.BytesIO(data)).pages[:25]
        return _clean(" ".join((p.extract_text() or "") for p in pages))[:80000]
    except Exception:
        return ""

def search_arxiv(kws, limit=20):
    def run(q):
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": 0, "max_results": limit, "sortBy": "relevance"})
        root = ET.fromstring(http_get(url))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in root.findall("a:entry", ns):
            title, abs_ = _clean(e.findtext("a:title", "", ns)), _clean(e.findtext("a:summary", "", ns))
            if title and abs_:
                out.append({"title": title, "abstract": abs_, "url": _clean(e.findtext("a:id", "", ns)),
                            "source": "arXiv · " + (e.findtext("a:published", "", ns) or "")[:4]})
        return out
    res = run(" AND ".join(f"all:{k}" for k in kws[:3]))
    if len(res) < 5:                                   # too strict -> widen
        res += run(" OR ".join(f"all:{k}" for k in kws[:6]))
    return res

def search_openalex(kws, limit=20):
    params = {"search": " ".join(kws[:8]), "per-page": limit,
              "select": "id,title,abstract_inverted_index,doi,publication_year,primary_location"}
    if os.getenv("APEX_MAILTO"):
        params["mailto"] = os.getenv("APEX_MAILTO")
    import json
    data = json.loads(http_get("https://api.openalex.org/works?" + urllib.parse.urlencode(params)))
    out = []
    for w in data.get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        if not inv or not w.get("title"):
            continue
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        loc = w.get("primary_location") or {}
        url = w.get("doi") or loc.get("landing_page_url") or w.get("id") or ""
        out.append({"title": _clean(w["title"]), "abstract": " ".join(x[1] for x in pos), "url": url,
                    "source": "OpenAlex · %s" % (w.get("publication_year") or "")})
    return out

def search_free_sources(kws, limit=12):
    """Extra open discovery sources. Failures are isolated so one API cannot block search."""
    import json
    q = " ".join(kws[:6])
    out = []
    try:
        d = json.loads(http_get("https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
            {"query": q, "limit": min(limit, 100), "fields": "title,abstract,url,year,openAccessPdf"})))
        for x in d.get("data", []):
            title, abstract = _clean(x.get("title")), _clean(x.get("abstract"))
            if title and abstract:
                oa = x.get("openAccessPdf") or {}
                out.append({"title": title, "abstract": abstract, "url": oa.get("url") or x.get("url", ""),
                            "source": "Semantic Scholar · " + str(x.get("year") or "")})
    except Exception:
        pass
    try:
        d = json.loads(http_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
            {"query": q, "format": "json", "pageSize": limit, "resultType": "core"})))
        for x in d.get("resultList", {}).get("result", []):
            title, abstract = _clean(x.get("title")), _clean(x.get("abstractText"))
            if title and abstract:
                pid = x.get("pmcid")
                out.append({"title": title, "abstract": abstract,
                            "url": "https://europepmc.org/articles/" + pid if pid else "https://europepmc.org/article/MED/" + str(x.get("id", "")),
                            "source": "Europe PMC · " + str(x.get("pubYear", ""))})
    except Exception:
        pass
    core_key = os.getenv("APEX_CORE_API_KEY", "").strip()
    if core_key:
        try:
            req = urllib.request.Request("https://api.core.ac.uk/v3/search/works?" + urllib.parse.urlencode({"q": q, "limit": limit}),
                                         headers={"Authorization": "Bearer " + core_key, "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read().decode("utf-8", errors="ignore"))
            for x in d.get("results", []):
                title, abstract = _clean(x.get("title")), _clean(x.get("abstract"))
                if title and abstract:
                    out.append({"title": title, "abstract": abstract, "url": x.get("downloadUrl") or x.get("sourceFulltextUrls", [""])[0],
                                "source": "CORE · " + str(x.get("yearPublished") or "")})
        except Exception:
            pass
    try:
        d = json.loads(http_get("https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query": q, "rows": limit, "select": "title,abstract,DOI,published-print,published-online"})))
        for x in d.get("message", {}).get("items", []):
            title = _clean((x.get("title") or [""])[0]); abstract = _clean(re.sub("<[^>]+>", " ", x.get("abstract", "")))
            if title and abstract:
                year = ((x.get("published-print") or x.get("published-online") or {}).get("date-parts") or [[""]])[0][0]
                out.append({"title": title, "abstract": abstract, "url": "https://doi.org/" + x.get("DOI", ""),
                            "source": "Crossref · " + str(year)})
    except Exception:
        pass
    return out

def live_search(text, title="", kws=None):
    """returns (candidates, errors)"""
    kws = kws or keywords(text, title)
    if len(kws) < 2:
        return [], []
    key = hashlib.md5(" ".join(kws).encode()).hexdigest()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1], []
    errors, results = [], []
    with ThreadPoolExecutor(4) as ex:
        futs = {"arXiv": ex.submit(search_arxiv, kws), "OpenAlex": ex.submit(search_openalex, kws),
                "Europe PMC/Crossref": ex.submit(search_free_sources, kws)}
        for name, f in futs.items():
            try:
                results += f.result(timeout=20)
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}")
    seen, uniq = set(), []
    for r in results:
        k = re.sub(r"\W+", "", r["title"].lower())
        if k not in seen:
            seen.add(k); uniq.append(r)
    if uniq:
        _cache[key] = (time.time(), uniq)
    return uniq, errors
