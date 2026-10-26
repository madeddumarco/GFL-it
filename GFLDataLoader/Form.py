import abc, os
from typing import Union 

class Form(abc.ABC):

    changed = None

    @abc.abstractmethod
    def __init__(self):
        pass

    def from_dict(dict_form: dict) -> "Form":
        form_type = dict_form["form_type"]
        if form_type == "mascfem":
            return FormMascFem(**dict_form)
        elif form_type == "unchanged":
            return FormUnchanged(**dict_form)
        elif form_type == "changed":
            return FormChanged(**dict_form)
        elif form_type == "neo":
            return FormNeo(**dict_form)
        else:
            raise Exception(f"type form: {form_type} not recognized")


    def get_dict(self):
        return {k:v for (k,v) in self.__dict__.items() if k not in ["act_extract", "sources"]}
    
    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass

    #method needed when we create the set of forms for forms_cv + forms_neo
    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    def eq_source(s1: dict, s2: dict) -> bool:
        attr_eq = ["sent_id", "span_id", "rew"]
        return all(s1[attr] == s2[attr] for attr in attr_eq)
    
    @classmethod
    def is_changed(cls) -> bool:
        if cls.changed is not None:
            return cls.changed
        else:
            raise Exception(f"Form class: {type(cls)} does not have 'changed' attribute")

    @abc.abstractmethod
    def is_same_rew(self, rewriting: Union[str, tuple]) -> bool:
        pass

#this class works only for CV
class FormMascFem(Form):

    changed = True

    def __init__(self, entire_masc: str = None, entire_fem: str = None, suffix_masc: str = None, suffix_fem: str = None, 
                 form_type: str = ""):
        self.entire_masc = entire_masc
        self.entire_fem = entire_fem
        self.suffix_masc = suffix_masc
        self.suffix_fem = suffix_fem
        self.form_type = "mascfem"
        self.set_none_fields()
        self.attr_to_save = ["entire_masc", "entire_fem", 
                             "suffix_masc", "suffix_fem", 
                             "form_type"]
    
    def __str__(self) -> str:
        #return f"fmf, m:{self.entire_masc}, f: {self.entire_fem}, s_m: {self.suffix_masc}, s_f: {self.suffix_fem}, \nsource_span: {self.source["tknzd_span"]} \nsource_rew: {self.source["rew"]} \n ---"
        return f"fmf, m:{self.entire_masc}, f: {self.entire_fem}, s_m: {self.suffix_masc}, s_f: {self.suffix_fem}\n ---"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, FormMascFem):
            return (self.entire_fem == other.entire_fem and
                    self.entire_masc == other.entire_masc and
                    self.suffix_fem == other.suffix_fem and
                    self.suffix_masc == other.suffix_masc)
        else:
            return False
    
    def __hash__(self) -> int:
        return hash((self.form_type, self.entire_masc, self.entire_fem))
    
    def set_attr_if_none(self, name_attr: str, val) -> None:
        if getattr(self, name_attr) is None:
            setattr(self, name_attr, val)

    
    #TODO TEST
    #method that given the current state sets all fields that are currently None 
    def set_none_fields(self) -> None:
        if (self.entire_masc is not None and self.entire_fem is not None):
            prefix = os.path.commonprefix([self.entire_masc, self.entire_fem])
            suffix_masc = self.entire_masc.removeprefix(prefix)
            suffix_fem = self.entire_fem.removeprefix(prefix)
            self.set_attr_if_none("suffix_masc", suffix_masc)
            self.set_attr_if_none("suffix_fem", suffix_fem)

        elif (self.entire_masc is not None and self.entire_fem is not None and self.suffix_fem is not None):
            prefix = self.entire_fem.removesuffix(self.suffix_fem)
            suffix_masc = self.entire_masc.removeprefix(prefix)
            self.set_attr_if_none("suffix_masc", suffix_masc)

        for field in ["entire_masc", "entire_fem", "suffix_masc", "suffix_fem"]:
            if getattr(self, field) is None:
                print(f"field: {field} is None")
                raise Exception


    def is_same_rew(self, rewriting: Union[str, str]) -> bool:
        if isinstance(rewriting, tuple):
            masc_form, fem_form = rewriting
            masc_form, fem_form = masc_form.strip(), fem_form.strip()
            return (masc_form.casefold() == self.entire_masc.casefold() and
                    fem_form.casefold() == self.entire_fem.casefold())
        else:
            return False

#this class works for both Neo and CV
class FormUnchanged(Form):

    changed = False

    def __init__(self, token_lower: str, form_type: str = "") -> None:
        #TODO: maybe the token_lower is not enough and we should have the cased token as well
        self.token_lower = token_lower
        self.form_type = "unchanged"
        self.attr_to_save = ["token_lower", 
                             "form_type"]
    def __eq__(self, other) -> bool:
        if isinstance(other, FormUnchanged):
            return (self.token_lower == other.token_lower)
        else:
            return False
        
    def __hash__(self) -> int:
        return hash((self.form_type, self.token_lower))
    
    def __str__(self) -> str:
        #return f"unchanged, t_l:{self.token_lower}, \nsource_span: {self.source["tknzd_span"]} \nsource_rew: {self.source["rew"]} \n ---"
        return f"unchanged, t_l:{self.token_lower}---"
    

    def is_same_rew(self, rewriting: Union[str, tuple]) -> bool:
        if isinstance(rewriting, str):
            rewriting = rewriting.strip()
            return rewriting.casefold() == self.token_lower.casefold()
        else: 
            return False

#this class works for both Neo and CV
class FormChanged(Form):

    changed = True

    def __init__(self, old_form: str, new_form: str, form_type: str = "") -> None:
        #TODO: da capire se old_form e new_form sono cased o lower
        self.old_form = old_form
        self.new_form = new_form
        self.form_type = "changed"
        self.attr_to_save = ["old_form", "new_form", 
                             "form_type"]
    def __eq__(self, other) -> bool:    
        if isinstance(other, FormChanged):
            return (self.old_form == other.old_form and 
                    self.new_form == other.new_form)
        else:
            return False
        
    def __hash__(self) -> int:
        return hash((self.form_type, self.old_form, self.new_form))
    
    def __str__(self) -> str:
        #return f"changed, old:{self.old_form}; new: {self.new_form}, \nsource_span: {self.source["tknzd_span"]} \nsource_rew: {self.source["rew"]} \n ---"
        return f"changed, old:{self.old_form}; new: {self.new_form} ---"
    

    def is_same_rew(self, rewriting: Union[str, tuple]) -> bool:
        if isinstance(rewriting, str):
            rewriting = rewriting.strip()
            return rewriting.casefold() == self.new_form.casefold()
        else:
            return False
    
#this class works only for Neo
class FormNeo(Form):

    changed = True

    def __init__(self, form_neo: str, form_type: str = "") -> None:
        #TODO: da capire casing di form_neo e se salvare anche una lista di neomorfermi che vanno bene assieme?
        self.form_neo = form_neo
        self.form_type = "neo"
        self.attr_to_save = ["form_neo",
                             "form_type"]

    def __eq__(self, other) -> bool:    
        if isinstance(other, FormNeo):
            return (self.form_neo == other.form_neo)
        else:
            return False
    
    def __hash__(self) -> int:
        return hash((self.form_type, self.form_neo))
        
    def __str__(self) -> str:
        #return f"changed, old:{self.old_form}; new: {self.new_form}, \nsource_span: {self.source["tknzd_span"]} \nsource_rew: {self.source["rew"]} \n ---"
        return f"neo, form_neo: {self.form_neo} ---"
    
    def is_same_rew(self, rewriting: Union[str, tuple]) -> bool:
        if isinstance(rewriting, str):
            rewriting = rewriting.strip()
            return rewriting.casefold() == self.form_neo.casefold()
        else:
            return False
        