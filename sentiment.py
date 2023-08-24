from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, Field


class Tags(BaseModel):
    sentiment: str = Field(..., enum=["positive", "neutral", "negative"])
    stars: int = Field(
        ...,
        description="describes how easily this email can be classified",
        enum=[1, 2, 3, 4, 5],
    )
    category: str = Field(
        ..., enum=["New Booking Creation", "Booking Amendment", "BL Creation", "BL Amendment", "Doc receipts", "Master Bol submission & validation", "Tally sheet", "pre-Alert documents"],
        description="A multi class classifier for the email provided, emails are related to shipping."
    )

load_dotenv()
schema = {
    "properties": {
        "sentiment": {"type": "string", "description": "Classify the email on the sentiment i.e. positive, negative or neutral"},
        "stars": {"type": "integer", "description": "describes how easily this email can be classified from 1 to 5"},
        "category": {"type": "string", "description": "Classify the email in following categories : 1. New Booking Creation 2. Booking Amendment 3. BL Creation 4. BL Amendment 5. Doc receipts 6. Master Bol submission & validation 7. Tally sheet 8. pre-Alert documents"},
    }
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
chain = create_tagging_chain(schema, llm)

review_01 = "Please amend cnee\nBest regards,\nAndrea Nava\nExport documentation \nEcu Worldwide Italy Srl\nVia Liguria, 5 \u2013 20068 Peschiera Borromeo (MI) - Italy\nD: +39 02 95656441\nE: andreanava@ecuworldwide.com\nW: www.ecuworldwide.com <http://www.ecuworldwide.com/> \n <https://www.facebook.com/ecuworldwide/>  <https://www.linkedin.com/company/ecu-worldwide/>  <https://twitter.com/ECUworldwide?s=20>  <https://www.youtube.com/channel/UC0oF9PD-wa8KqNCa1vqq7NA> \nFrom: Paolo Melandri <paolo.melandri@si-log.com> \nSent: mercoled\u00ec 8 marzo 2023 11:45\nTo: documentationitaly (IT-MIL ECU Worldwide) <documentationitaly@ecuworldwide.com>\nCc: Follow Up (IT-MIL ECU Worldwide) <followupmilan@ecuworldwide.com>\nSubject: R: MIL/MIA/1917321\n\nHello: no : sorry : pls consider cnee as follows: \nMIAMI BEEF CO., INC.\n4870 NW 157th ST.\nMIAMI LAKES, FL 33014\nSTATI UNITI\nPaolo Melandri\nOperations Team\nSOLEIL INTERNATIONAL S.P.A.\nVia Guicciardini,13 | 50125 Firenze (Italy) \nP.IVA 06074470482 - Codice SDI RR66BDG\nPhone: +39 055-2381997\nDa: documentationitaly (IT-MIL ECU Worldwide) <documentationitaly@ecuworldwide.com <mailto:documentationitaly@ecuworldwide.com> > \nInviato: Wednesday, March 8, 2023 11:28 AM\nA: Paolo Melandri <paolo.melandri@si-log.com <mailto:paolo.melandri@si-log.com> >\nCc: Follow Up (IT-MIL ECU Worldwide) <followupmilan@ecuworldwide.com <mailto:followupmilan@ecuworldwide.com> >\nOggetto: RE: MIL/MIA/1917321\nDear,\nFor the subject mentioned booking while Processing AMS for Miami \nBelow AMS details shows consignee from Canada\nKindly confirm the correct Consignee details \nThanks & Regards,\nTejas."

print(chain.run(review_01))

review_02 = """\nHi Team, \nPlease arrange a booking for the below: \nFrom/To:  AUSYD / NZAKL\nQuantity: 6 PLT\nVolume: 9.084 CBM\nWeight: 1800 Kg\nGood description: EARTHENWARE, BONE CHINA AND CRYSTAL TABLEWARE\nThanks & Best regards,\nVincent\nAs agents for and on behalf of Pyramid Lines Singapore\nVincent Huang\nOcean Export Operator \nVincent.Huang@Cevalogistics.com \nTel: +61 2 7226 9063\nUnit 10B, 1801 Botany Road, Banksmeadow, NSW 2019, Australia \nwww.cevalogistics.com <http://www.cevalogistics.com/>  \nNVOCC Services are provided by CEVA as agents for Pyramid Lines\nSingapore Pte. Ltd.\nThis e-mail message is intended for the above named recipient(s) only.\nIt may contain confidential information that is privileged. If you are\nnot the intended recipient, you are hereby notified that any\ndissemination, distribution or copying of this e-mail and any\nattachment(s) is strictly prohibited. If you have received this e-mail\nby error, please immediately notify the sender by replying to this\ne-mail and deleting the message including any attachment(s) from your\nsystem. Thank you in advance for your cooperation and assistance.\nAlthough the company has taken reasonable precautions to ensure no\nviruses are present in this email, the company cannot accept\nresponsibility for any loss or damage arising from the use of this email\nor attachments."""

print(chain.run(review_02))

schema = {
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "stars": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "describes how easily this email can be classified",
        },
        "category": {
            "type": "string",
            "enum": ["New Booking Creation", "Booking Amendment", "BL Creation", "BL Amendment", "Doc receipts", "Master Bol submission & validation", "Tally sheet", "pre-Alert documents"],
            "description": "A multi class classifier for the email provided, emails are related to shipping.",
        },
    },
    "required": ["category", "sentiment", "stars"],
}

chain = create_tagging_chain(schema, llm)
print("\n------------AFTER SCHEMA UPDATE--------------\n")
print(chain.run(review_01))
print(chain.run(review_02))

print("\n----------Pydantic output------------------\n")
chain = create_tagging_chain_pydantic(Tags, llm)
res = chain.run(review_01)
print(type(res))