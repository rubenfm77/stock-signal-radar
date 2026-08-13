"""
Curated ticker universe across US, European, and Spanish markets.
Kept intentionally small (~60-80 names) for fast iteration; expand once
the pipeline is validated. yfinance ticker suffix conventions:
  - Spain (Madrid):   .MC
  - Germany (Xetra):  .DE
  - France (Paris):   .PA
  - Netherlands:      .AS
  - Italy (Milan):    .MI
  - US:               no suffix
"""

IBEX35 = [
    "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "TEF.MC", "REP.MC",
    "FER.MC", "AMS.MC", "CLNX.MC", "ELE.MC", "NTGY.MC", "CABK.MC",
    "AENA.MC", "MAP.MC", "ACS.MC", "GRF.MC", "SAB.MC", "IAG.MC",
    "COL.MC", "MRL.MC",
]

EUROSTOXX_SAMPLE = [
    "ASML.AS", "SAP.DE", "SIE.DE", "ALV.DE", "MC.PA", "OR.PA",
    "TTE.PA", "AIR.PA", "SU.PA", "DTE.DE", "BAS.DE", "ADS.DE",
    "ENEL.MI", "ISP.MI", "STLAM.MI",
]

US_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "PG", "MA", "HD", "COST", "AVGO",
    "ADBE", "NFLX", "CRM", "AMD", "INTC", "PEP", "KO",
]

UNIVERSE = {
    "IBEX35 (España)": IBEX35,
    "EuroStoxx (muestra)": EUROSTOXX_SAMPLE,
    "US (muestra S&P/Nasdaq)": US_SAMPLE,
}


def all_tickers() -> list[str]:
    seen, out = set(), []
    for group in UNIVERSE.values():
        for t in group:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
