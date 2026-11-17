"""A stand-in for the model provider (MINISAKIAS_DEMO=1): the website, the search and the citations work without
an API key, and the answer says plainly that it is not a real one. For trying the site locally, never for citizens."""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

_LABEL = re.compile(r"^\[([^\]]+)\]", re.M)


class DemoLLM:
    async def complete_json(self, system: str, messages: list[dict], **_) -> dict:
        raise ValueError("demo mode has no router; the search uses the question's own words")

    async def stream(self, system: str, messages: list[dict], **_) -> AsyncIterator[str]:
        labels = _LABEL.findall(messages[-1]["content"])[:4]
        cited = "\n".join(f"- [{label}]" for label in labels) or "- (η αναζήτηση δεν βρήκε σχετικά άρθρα)"
        text = (
            "## Λειτουργία επίδειξης\n\n"
            "Αυτή **δεν είναι πραγματική απάντηση**: η υπηρεσία τρέχει χωρίς μοντέλο τεχνητής νοημοσύνης "
            "(`MINISAKIAS_DEMO=1`), για να δείτε πώς λειτουργεί ο ιστότοπος.\n\n"
            "Η αναζήτηση στη συλλογή νόμων όμως είναι πραγματική. Για το μήνυμά σας βρήκε, μεταξύ άλλων:\n\n"
            f"{cited}\n\n"
            "Πατήστε σε ένα άρθρο για να διαβάσετε το κείμενό του. Με ρυθμισμένο κλειδί παρόχου, εδώ θα βλέπατε "
            "τι σημαίνει νομικά το πρόβλημά σας, ποιους δρόμους έχετε και ποιες προθεσμίες τρέχουν.\n"
        )
        for i in range(0, len(text), 24):
            await asyncio.sleep(0.02)
            yield text[i: i + 24]
