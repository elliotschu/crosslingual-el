"""
Elliot Schumacher, Johns Hopkins University
Created 10/15/19
"""
import re
import logging
import pprint

class Mentions(object):
    def __init__(self):
        self.entities = []
        self.docs = {}


class Mention_Entity(object):
    def __init__(self, id, mention, kbid, doc_filename, doc, doc_id, kb_type=None, lang = None):
        self.id = id
        self.doc_id = doc_id
        self.mention = mention
        self.kbid = kbid
        self.doc = doc
        self.kb_type = kb_type
        self.index = None
        self.nil = False
        self.doc_filename = doc_filename
        self.candidate_kb = {}
        self.lang = lang
        self.cand_skip = False

        self.mention_start = mention.start_char
        self.mention_end = mention.end_char

    def add_candidate(self, kbid, score):
        if kbid not in self.candidate_kb:
            self.candidate_kb[kbid] = score
    def __str__(self):
        return f"id:{self.id}\tname:{self.mention}"

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['mention']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        self.mention = self.doc.char_span(self.mention_start, self.mention_end )
        if self.mention is None:
            raise AssertionError(f"{str(self)} mention is None")
class Ontology(object):
    def __init__(self):
        self.entities = {}
        #self.pp = pprint.PrettyPrinter(indent=4)
        self.max_name_len = 50
        self.entity_mappings = ['rs_name',
                                "rs_label",
                                'f_type.object.name',
                                'f_common.topic.alias',
                                'fk_key.en',
                                'fk_key.wikipedia.en',
                                'fk_key.wikipedia.en_title',
                                'f_base.schemastaging.context_name.official_name',
                                'f_base.schemastaging.context_name.nickname',
                                'f_base.arthist.helynevek.helynev',
                                'f_base.aareas.schema.administrative_area.adjectival_form',
                                'f_base.arthist.int_zm_nyek.intezmenynev',
                                'f_base.schemastaging.context_name.short_name',
                                'f_common.document.text',
                                'f_common.topic.description',
                                'f_base.kwebbase.kwtopic.assessment',]
        self.exclude_fields = ['f_base.biblioness.bibs_location.loc_type',
                               'f_common.identity.daylife_topic',
                               'f_book.author.openlibrary_id',
                               'f_base.foursquare.foursquare_location.foursquare_id',
                               '_doc_id',
                               '_score']
        self.entity_mappings_text = ['f_common.topic.description',
                                     'f_common.document.text',
                                     ]
    def get_text(self, ent):
        if ent.from_wikipedia:
            ent.text = ent.fields['article']
            return ent.text
        else:
            if ent.text is None:
                for field_title in self.entity_mappings_text:
                    if field_title in ent.fields:
                        if type(ent.fields[field_title]) is list:
                            possible_names = []
                            for field_item in ent.fields[field_title]:
                                pn = self.process_name(field_item)
                                if pn is not None:
                                    possible_names.append(pn)
                            if len(possible_names) > 0:
                                possible_names = sorted(possible_names, key=lambda x: len(x), reverse=True)
                                ent.text = possible_names[0]

                        else:
                            ent.text = self.process_name(ent.fields[field_title])
            if ent.text is None:

                potential_names = []
                for k in ent.fields.keys():
                    if k not in self.entity_mappings and k not in self.exclude_fields:
                        if type(ent.fields[k]) is list:
                            for field_item in ent.fields[k]:
                                cand_name = self.process_name_backup(field_item)
                                if cand_name is not None:
                                    potential_names.append((cand_name, k))
                        elif type(ent.fields[k]) is str:
                            cand_name = self.process_name_backup(ent.fields[k])
                            if cand_name is not None:
                                potential_names.append((cand_name, k))
                if len(potential_names) > 0:
                    potential_names = sorted(potential_names, key=lambda x: len(x), reverse=True)
                    ent.text = potential_names[0][0]
            if ent.text is None:
                ent.text = ent.fields["subject"]
            return ent.text


    def get_text_l2(self, ent,  lang_id = "zh"):
        if ent.from_wikipedia:
            ent.text = ent.fields['article']
            return ent.text
        else:
            if ent.text is None:
                for field_title in self.entity_mappings_text:
                    if field_title in ent.fields:
                        if type(ent.fields[field_title]) is list:
                            possible_names = []
                            for field_item in ent.fields[field_title]:
                                pn = self.process_name_l2(field_item, lang_id)
                                if pn is not None:
                                    possible_names.append(pn)
                            if len(possible_names) > 0:
                                possible_names = sorted(possible_names, key=lambda x: len(x), reverse=True)
                                ent.text = possible_names[0]

                        else:
                            ent.text = self.process_name_l2(ent.fields[field_title], lang_id)
            if ent.text is None:

                potential_names = []
                for k in ent.fields.keys():
                    if k not in self.entity_mappings and k not in self.exclude_fields:
                        if type(ent.fields[k]) is list:
                            for field_item in ent.fields[k]:
                                cand_name = self.process_name_backup_l2(field_item, lang_id)
                                if cand_name is not None:
                                    potential_names.append((cand_name, k))
                        elif type(ent.fields[k]) is str:
                            cand_name = self.process_name_backup_l2(ent.fields[k], lang_id)
                            if cand_name is not None:
                                potential_names.append((cand_name, k))
                if len(potential_names) > 0:
                    potential_names = sorted(potential_names, key=lambda x: len(x), reverse=True)
                    ent.text = potential_names[0][0]
            # if ent.text is None:
            #     ent.text = ent.fields["subject"]
            return ent.text


    def process_name(self, item_string):
        name = None
        non_eng = False
        l2_names = {}
        if "@" in item_string:
            try:
                name_group = re.match(r"^\"(.*)\"@(.*)$", item_string)
                lang_id = name_group.group(2)
                if lang_id == "en":
                    name = name_group.group(1)
                else:
                    non_eng = True
                    l2_names[lang_id] = name_group.group(1)
            except:
                pass
        if not non_eng and name is None:
            name = item_string.replace("\"", "")
        return name

    def process_name_backup(self, item_string):
        name = None
        non_eng = False
        l2_names = {}

        if "@" in item_string:
            try:
                name_group = re.match(r"^\"(.*)\"@(.*)$", item_string)
                lang_id = name_group.group(2)
                if lang_id == "en":
                    name = name_group.group(1)
                else:
                    non_eng = True
                    l2_names[lang_id] = name_group.group(1)

            except:
                pass
        if not non_eng and name is None:
            try:
                name_group = re.match(r"^\"(.*)\"$", item_string)
                name = name_group.group(1)
            except:
                pass
        return name

    def get_name(self, field_dict):
        name = None
        for name_field in self.entity_mappings:
            if name_field in field_dict:
                if type(field_dict[name_field]) is list:
                    possible_names = []
                    for field_item in field_dict[name_field]:
                        pn = self.process_name(field_item)
                        if pn is not None:
                            possible_names.append(pn)
                    if len(possible_names) > 0:
                        possible_names = sorted(possible_names, key=lambda x: len(x), reverse=True)
                        name = possible_names[0]

                else:
                    name = self.process_name(field_dict[name_field])
            if name is not None:
                break
        if name is None:
            potential_names = []
            for k in field_dict.keys():
                if k not in self.entity_mappings and k not in self.exclude_fields:
                    if type(field_dict[k]) is list:
                        for field_item in field_dict[k]:
                            cand_name = self.process_name_backup(field_item)
                            if cand_name is not None:
                                potential_names.append((cand_name, k))
                    else:
                        cand_name = self.process_name_backup(field_dict[k])
                        if cand_name is not None:
                            potential_names.append((cand_name, k))
            if len(potential_names) > 0:
                potential_names = sorted(potential_names, key=lambda x: len(x), reverse=True)
                name = potential_names[0]
        if name is None:
            name = field_dict["subject"]
        return name

    def process_name_l2(self, item_string, pref_lang_id):
        name = None
        if "@" in str(item_string):
            try:
                name_group = re.match(r"^\"(.*)\"@(.*)$", item_string)
                lang_id = name_group.group(2)
                if lang_id == pref_lang_id:
                    name = name_group.group(1)

            except:
                pass
        return name

    def process_name_backup_l2(self, item_string, pref_lang_id):
        name = None

        if "@" in str(item_string):
            try:
                name_group = re.match(r"^\"(.*)\"@(.*)$", item_string)
                lang_id = name_group.group(2)
                if lang_id == pref_lang_id:
                    name = name_group.group(1)

            except:
                pass

        return name

    def get_name_l2(self, field_dict, lang_id = "zh"):
        name = None
        for name_field in self.entity_mappings:
            if name_field in field_dict:
                if type(field_dict[name_field]) is list:
                    possible_names = []
                    for field_item in field_dict[name_field]:
                        pn = self.process_name_l2(field_item, lang_id)
                        if pn is not None:
                            possible_names.append(pn)
                    if len(possible_names) > 0:
                        possible_names = sorted(possible_names, key=lambda x: len(x), reverse=True)
                        name = possible_names[0]

                else:
                    name = self.process_name_l2(field_dict[name_field], lang_id)
            if name is not None:
                break
        if name is None:
            potential_names = []
            for k in field_dict.keys():
                if k not in self.entity_mappings and k not in self.exclude_fields:
                    if type(field_dict[k]) is list:
                        for field_item in field_dict[k]:
                            cand_name = self.process_name_backup_l2(field_item, lang_id)
                            if cand_name is not None:
                                potential_names.append((cand_name, k))
                    else:
                        cand_name = self.process_name_backup_l2(field_dict[k], lang_id)
                        if cand_name is not None:
                            potential_names.append((cand_name, k))
            if len(potential_names) > 0:
                potential_names = sorted(potential_names, key=lambda x: len(x), reverse=True)
                name = potential_names[0]
        # if name is None:
        #     name = field_dict["subject"]
        return name

    def add(self, id, field_dict, add_if_no_name = True):
        if id not in self.entities:
            ent_type = []
            if "r_type" in field_dict:
                if type(field_dict["r_type"]) is list:
                    ent_type = field_dict["r_type"]
                else:
                    ent_type = [field_dict["r_type"]]
            name = self.get_name(field_dict)
            oe = Ontology_Entity(id, name, ent_type, field_dict)

            for lang_id in ["zh", "es"]:
                l2_name = self.get_name_l2(field_dict, lang_id)
                if l2_name is not None:
                    oe.l2_names[lang_id] = l2_name
                l2_text = self.get_text_l2(oe, lang_id)
                if l2_text is not None:
                    oe.l2_text[lang_id] = l2_text
            if not add_if_no_name and name is None:
                return False
            self.entities[id] = oe
        return True

    def add_wiki(self, id, field_dict, add_if_no_name = True):
        if id not in self.entities:
            ent_type = []
            if "categories" in field_dict:
                if type(field_dict["categories"]) is list:
                    ent_type = field_dict["categories"]
                else:
                    ent_type = [field_dict["categories"]]
            name = field_dict['title']
            oe = Ontology_Entity(id, name, ent_type, field_dict, from_wikipedia=True)
            self.entities[id] = oe

class Ontology_Entity(object):
    def __init__(self, id, name, ent_type, fields, from_wikipedia=False):
        self.id = id
        self.fields = fields
        self.name = name
        self.l2_names = {}
        self.l2_text = {}

        self.type = ent_type
        self.index = None
        self.text = None
        self.from_wikipedia = from_wikipedia
        self.popularity = 0

    def __str__(self):
        return "id:{0}\tname:{1}\tfields{2}".format(self.id, self.name, self.fields)
