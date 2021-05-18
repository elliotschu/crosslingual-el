#!/usr/bin/python

"""
Entity Linking Scoring Script

version 0.7:
September 22, 2013
By Hoa Dang

Updated scorer to accept multiple links per query, as allowed in TAC KBP 2013.
Each line in the entity linking system output file consists of 2 or three space-delimited fields:
Field 1: query ID
Field 2: link to a KB (or NIL) ID
Field 3: confidence score (optional, default value is 1.0)

Consider only the system link with the highest confidence value for each element (aka query or mention) and compute scores as in version 0.6.

Implementation notes:
Incorporated getMap() function from previous versions into the readLinking() function.
Each item in el2kbid is now a list of KB_IDs (with associated confidence) rather than a single KB_ID.


version 0.6:
December 17, 2011
By Hoa Dang (hoa.dang@nist.gov)

1. Changed the header of the output table (replace "B^3" with "B^3+";
and replace "B^2" with "B^3").

2. Changed the way that scores are computed for subsets of queries from
the complete eval data, to allow a more meaningful comparison of the
same system across different subsets of the eval data.

In v0.4, to compute the B^3 F-score for a set of queries Q1 that is a
subset of the entire set of queries Q in the eval data, all queries
except for those in Q1 are removed from both the gold standard and
system submission before computing query-level precision and recall.
However, this biases the comparison of the same system's performance
across different subsets (specifically, between Q1 vs Q), because of
the different number of queries involved in the computation of
query-level precision.
Let {Q1, Q2, ... ,Qn} be a partition of Q.  Then for any Qi in the
partition, and any query q in Qi, the query-level precision and recall
of q is such that:
    B^3 Recall(q|Q) == B^3 Recall(q|Qi)
    B^3 Precision(q|Q) <= B^3 Precision(q|Qi).
That is, the query-level precision of q given a universe containing
only queries of type i, is generally higher than the query-level
precision of q given a universe containing queries of type i, plus
additional queries; the query-level Recall of q is the same in both
universes.  Therefore, each query q generally contributes less to
overall B-cubed F-score in the context of Q than in the context of any
subset of Q; and it's possible for the B^3 F-score of Q to be lower
than the B^3 F-score of *every* element in a partition of Q (when
the B^3 F-scores of the elements in the partition are very similar).

V0.6 supports comparison of a system's performance on different
subsets of an eval set Q by computing query-level precision and recall
with respect to the entire eval set Q.
The v0.6 scorer requires specification of both the "complete"
gold-standard linking, and gold standard linkings for a "focus" subset
of queries. The v0.6 scorer:
 (1) Computes [B^3 or B^3+] precision and recall for each query in the
     complete gold standard, within the environment defined by all
     queries in the complete gold standard;

 (2) Computes *average* precision and recall for the set of *focus*
     queries by taking the average of the query-level precision and
     recall of the focus queries.
 (3) Computes [B^3 or B^3+] F-score from the average precision and
     recall in (2)



version 0.5:
October 14, 2011
By Hoa Dang (hoa.dang@nist.gov)
Also log.info standard B^3 scores (sometimes called b2 in this implementation)

version 0.4:
September 30, 2011
By Hoa Dang (hoa.dang@nist.gov)
Allow queries in gold standard to be a subset of the queries in the
system output; discard all queries in the system output that are not
in the gold standard.  In this way, scores can be broken down by
different query types.

version 0.3
September 14, 2011
By Hoa Dang (hoa.dang@nist.gov)
Fixed computation of kbp2010_microaverage function to treat all NIL
labels as equivalent.
Added F1 score as official KBP 2011 score.

version 0.2:
May 11, 2011
By Javier Artiles (CUNY, javart@gmail.com)
Fixed bug in 'sameLinking' function (thanks to Timothy Nyberg) that
prevented NIL elements with different identifiers on the gold std. and
sys output sides from matching

version 0.1
March 2, 2011
By Javier Artiles (CUNY, javart@gmail.com)
Initial script.
In KBP2011, we have a new requirement for entity linking, which is to
cluster NIL queries. For a set of query names with source documents,
an entity linking system is required to: (1) judge whether each query
can be linked to any KB node; (2) Cluster all queries with NIL KB
entries into clusters.  Ultimately the system output can be viewed as
a collection of various clusters; some clusters are labeled as KB node
IDs. At the same time the answer key can also be viewed as a different
collection of clusters.  Therefore we will apply a modified B-Cubed
metric (called B-Cubed+) to evaluate these clusters.

"""

import re
import sys
import os
import logging
log = logging.getLogger()
from collections import defaultdict
class SystemOutputException(Exception): pass

def readLinking(goldStdFile):
    """
    Reads a file containing Entity Linking output according to the KBP 2013 format.
    Each line in the systeming output file consists of 2 or 3 fields:
      mention_id KB_ID (optional confidence, default is 1.0)
    Each line in the gold standard file may contain 2 or more than 3 fields, but we ignore all but the first 2 fields:
      mention_id KB_ID
    
    Do NOT change IDs in any way (i.e., do not Uppercase all IDs)
    Returns a pair of dictionaries:
        linking: KB_ID -> set of query_IDs
        el2kbid: query_ID -> list of pairs of KB_ID (or NIL ID) and confidence, highest confidence first
    """
    linking = dict()
    el2kbid = dict()
    for line in open(goldStdFile):
        d = re.split("\s+", line.strip())
        if len(d)<2:
            continue
        mention = d[0]
        kb_id   = d[1]
        if(len(d)==3):  # a system linking
            conf = d[2]
        else:
            conf = 1.0 

        if mention in el2kbid.keys():  # more than one link for this mention; put the earliest highest confidence link in front, and add additional links to back of list
            links = el2kbid[mention]   # list of (kb_id,conf) tuples
            bestlink, bestconf = links[0]
            if conf > bestconf:
                links.insert(0, (kb_id, conf))
            else:
                links.append((kb_id, conf))
        else:
            el2kbid[mention] = [(kb_id, conf)]
            

        if kb_id in linking.keys():
            linking[kb_id].add(mention) 
        else:
            linking[kb_id] = set([mention])  

    return (linking, el2kbid)


def best_linking (linking, el2kbid):
    """
    return a subset of the linking consisting of only the highest confidence link for each element in el2kbid
    """
    new_linking = dict()
    for m in el2kbid.keys():
        links = el2kbid[m]
        kb_id, conf = links[0] # highest confidence kb_id
        
        if kb_id in new_linking.keys():
            new_linking[kb_id].add(m)
        else:
            new_linking[kb_id] = set([m])
    return new_linking

def b2_correctness(el_a, el_b, system_el2kbid, gold_el2kbid):
    """
    A pair of elements (el_a, el_b) is considered 'correct' when they
    share the same cluster in the gold standard and
    share the same cluster in the system output
    Caller ensures that inSameSet(el_a, el_b, system_el2kbid)
    """
    correct = False

    if(inSameSet(el_a, el_b, system_el2kbid) and 
       inSameSet(el_a, el_b, gold_el2kbid)
       ):
        correct = True

    return correct

def b3_correctness(el_a, el_b, system_el2kbid, gold_el2kbid):
    """
    A pair of elements (el_a, el_b) is considered 'correct' when they
    share the same cluster in the gold standard and
    share the same cluster in the system output and
    have the same linking (see 'sameLinking' documentation).
    Caller ensures that inSameSet(el_a, el_b, system_el2kbid).
    """
    correct = False

    if(inSameSet(el_a, el_b, system_el2kbid) and 
       inSameSet(el_a, el_b, gold_el2kbid) and
       sameLinking(el_a, el_b, system_el2kbid, gold_el2kbid)  #THIS CONDITION DEPARTS FROM THE ORIGINAL BCUBED (extension for the Entity Linking problem)
       ):
        correct = True

    return correct

def sameLinking(el_a, el_b, system_el2kbid, gold_el2kbid):
    """
    For two elements (el_a, el_b) that are be mapped to the Knowledge Base in the gold standard
    'same linking' means that they share the same identifier, both in the gold standard and the system output.
    Elements that cannot be mapped to the KB are assigned a NILXXX identifier, where XXX is a string with no white space;
    for those elements 'same linking' is satisfied when the element's label begins with 'NIL' in both gold
    standard and system output. That means that we check that the elements have been correctly identified
    as not mapped to the KB, but ignore the particular identifier assigned on the gold std. and sys. output sides.
    """

    sys_alinks = system_el2kbid[el_a]
    sys_blinks = system_el2kbid[el_b]
    gol_alinks = gold_el2kbid[el_a]
    gol_blinks = gold_el2kbid[el_b]

    sys_el_a_id, sys_aconf = sys_alinks[0]
    sys_el_b_id, sys_bconf = sys_blinks[0]
    gol_el_a_id, gol_aconf = gol_alinks[0]
    gol_el_b_id, gol_bconf = gol_blinks[0]

    if sys_el_a_id.startswith('NIL'): sys_el_a_id = 'NIL'
    if sys_el_b_id.startswith('NIL'): sys_el_b_id = 'NIL'
    if gol_el_a_id.startswith('NIL'): gol_el_a_id = 'NIL'
    if gol_el_b_id.startswith('NIL'): gol_el_b_id = 'NIL'

    #log.info system_el2kbid
    
    return sys_el_a_id == sys_el_b_id == gol_el_a_id == gol_el_b_id

def inSameSet(el_a, el_b, el2kbid):
    alinks = el2kbid[el_a]
    blinks = el2kbid[el_b]
    akb_id, aconf = alinks[0]
    bkb_id, bconf = blinks[0]
    return akb_id == bkb_id  # only look at the highest confidence link 
#    return el2kbid[el_a] == el2kbid[el_b]

def b2_recall(system_output, gold_standard, sys_el2kbid, gold_el2kbid):
    return b2_precision(gold_standard, system_output, gold_el2kbid, sys_el2kbid)

def b3_recall(system_output, gold_standard, sys_el2kbid, gold_el2kbid):
    return b3_precision(gold_standard, system_output, gold_el2kbid, sys_el2kbid)
    

def b3_precision(system_output, gold_standard, sys_el2kbid, gold_el2kbid):
    """
    Extension of the original BCubed clustering metric as described by Daniel Bikel, Vittorio Castelli and Radu Florian in their technical report.
    """
    Prec = dict()
    
    for kb_id in system_output.keys():
        mention_set = system_output[kb_id]

        for el_a in mention_set:
            num_correct = 0
            
            for el_b in mention_set:
                correct = b3_correctness(el_a, el_b, sys_el2kbid, gold_el2kbid)
                if(correct): num_correct +=1

            el_pre = num_correct / float(len(mention_set))
            Prec[el_a] = el_pre
            
    return Prec


def b2_precision(system_output, gold_standard, sys_el2kbid, gold_el2kbid):
    """
    Implementation of the original BCubed clustering metric.
    Each element in the set of mentions is weighted equally.
    
    """
    Prec = dict()
    
    for kb_id in system_output.keys():
        mention_set = system_output[kb_id]

        for el_a in mention_set:
            num_correct = 0
            
            for el_b in mention_set:
                correct = b2_correctness(el_a, el_b, sys_el2kbid, gold_el2kbid)
                if(correct): num_correct +=1

            el_pre = num_correct / float(len(mention_set))
            Prec[el_a] = el_pre
            
    return Prec

"""
getMap is no longer used, functionality is incorporated in readLinking and filtersubset
"""
def getMap(linking):
    el2kbid = dict()
    for kbid in linking.keys():
        mentions = linking[kbid]
        for m in mentions:
            el2kbid[m] = kbid
    return el2kbid

def filtersubset(linking, el2kbid, gold_els):
    """
    remove from linking and rom el2kbid all mentions that are not in gold_els
    """
    newlinking = dict()
    new_el2kbid = dict()

    for kbid in linking.keys():
        mentions = linking[kbid]
        for m in mentions:
            if m in gold_els:
                if kbid in newlinking.keys():
                    newlinking[kbid].add(m)
                else:
                    newlinking[kbid] = set([m])

    for m in el2kbid.keys():
        if m in gold_els:
            new_el2kbid[m] = el2kbid[m]

    return (newlinking, new_el2kbid)
                

def kbp2010_microaverage(sys_el2kbid, gold_el2kbid):
    num_samples = len(gold_el2kbid.keys())
    num_correct_samples = 0

    for el in gold_el2kbid.keys():
        gold_links = gold_el2kbid[el]
        gold_kbid, gold_conf = gold_links[0]
        if gold_kbid.startswith('NIL'): gold_kbid = 'NIL'
        sys_links = sys_el2kbid[el]
        sys_kbid, sys_conf = sys_links[0]
        if sys_kbid.startswith('NIL'): sys_kbid = 'NIL'

        if gold_kbid == sys_kbid: num_correct_samples += 1

    #log.info num_correct_samples
    #log.info num_samples
    
    return num_correct_samples / float(num_samples)


def systemsRankingScript(goldStdFile, systemsDir, focusElFile):

    gold_standard, gold_el2kbid = readLinking(goldStdFile)
#    gold_el2kbid = getMap(gold_standard)
    gold_els = frozenset(gold_el2kbid.keys())

    focus_linking, focus_el2kbid = readLinking(focusElFile)
#    focus_el2kbid = getMap(focus_linking)
    focus_els = frozenset(focus_el2kbid.keys())
    
    log.info ("%d queries evaluated in focus subset: %s" % (len(focus_els), focusElFile))
    log.info ("%d queries in gold standard environment: %s" % (len(gold_els), goldStdFile))
    log.info ("system\tKBP2010 micro-average\tB^3 Precision\tB^3 Recall\tB^3 F1\tB^3+ Precision\tB^3+ Recall\tB^3+ F1") 

    runs = os.listdir(systemsDir)
    runs.sort(key=str.lower)
    all_results = defaultdict(lambda: [])
    for systemOutFile in runs:
        
        system_output, sys_el2kbid = readLinking(systemsDir+"/"+systemOutFile)
#        sys_el2kbid = getMap(system_output)

        #Make sure that the system output includes all and only items in the reference
        sys_els = frozenset(sys_el2kbid.keys())
        diff = gold_els - sys_els
        try:
            if len(diff) > 0:
                raise SystemOutputException
        except SystemOutputException:
            log.info ("[ERROR] The output in \""+systemsDir+systemOutFile+"\" is missing the following queries ("+str(', '.join(diff))+"). This system output wont be evaluated.")
            continue

        #Discard items in system output that are not in the reference
        diff = sys_els - gold_els
        try:
            if len(diff) > 0:
                raise SystemOutputException
        except SystemOutputException:
            #log.info "[ERROR] The output in \""+systemsDir+systemOutFile+"\" contains queries not present in the gold standard ("+str(', '.join(diff))+").  These extra queries will be ignored in the evaluation."
            system_output, sys_el2kbid = filtersubset(system_output, sys_el2kbid, gold_els)
#            sys_el2kbid = getMap(system_output)
            sys_els = frozenset(sys_el2kbid.keys())

            #log.info "Number of queries in filtered system output:  %d" % (len(sys_els))


        # We'll consider only the highest confidence link for each
        # element in the system linking for now.  (The gold standard
        # linking only has one link per element, so no need to do
        # anything with it for now)
            
        system_output = best_linking(system_output, sys_el2kbid)
            
        system_name = systemOutFile

        kbp_micro_aver = kbp2010_microaverage(sys_el2kbid, focus_el2kbid)

        b2_pre = b2_precision(system_output, gold_standard, sys_el2kbid, gold_el2kbid)
        b2_rec = b2_recall(system_output, gold_standard, sys_el2kbid, gold_el2kbid)

        #b3_pre = b3_precision(system_output, gold_standard, sys_el2kbid, gold_el2kbid)
        #b3_rec = b3_recall(system_output, gold_standard, sys_el2kbid, gold_el2kbid)

        el_pre_sums = 0.0
        el_rec_sums = 0.0
        el_f_sums = 0.0
#         #for el_a in focus_els:
#             #el_pre_sums += b3_pre[el_a]
#             #el_rec_sums += b3_rec[el_a]
# #            log.info "%s\t%.2f\t%.2f" % (el_a, b3_pre[el_a] , b3_rec[el_a])
#             #if (b3_pre[el_a] + b3_rec[el_a] > 0):
#             #    el_f_sums += 2 * b3_pre[el_a] * b3_rec[el_a] / (b3_pre[el_a] + b3_rec[el_a])
#         b3_pre_avg = el_pre_sums / float (len(focus_els))
#         b3_rec_avg = el_rec_sums / float (len(focus_els))
#         b3_avg_f = el_f_sums / float (len(focus_els))
#         if (b3_pre_avg + b3_rec_avg > 0):
#             f_b3 = 2 * b3_pre_avg * b3_rec_avg / (b3_pre_avg + b3_rec_avg)
#         else:
#             f_b3 = 0.0

        el_pre_sums = 0.0
        el_rec_sums = 0.0
        el_f_sums = 0.0
        for el_a in focus_els:
            el_pre_sums += b2_pre[el_a]
            el_rec_sums += b2_rec[el_a]
#            log.info "%s\t%.2f\t%.2f" % (el_a, b2_pre[el_a] , b2_rec[el_a])
            if (b2_pre[el_a] + b2_rec[el_a] > 0):
                el_f_sums += 2 * b2_pre[el_a] * b2_rec[el_a] / (b2_pre[el_a] + b2_rec[el_a])
        b2_pre_avg = el_pre_sums / float (len(focus_els))
        b2_rec_avg = el_rec_sums / float (len(focus_els))
        b2_avg_f = el_f_sums / float (len(focus_els))
        if (b2_pre_avg + b2_rec_avg > 0):
            f_b2 = 2 * b2_pre_avg * b2_rec_avg / (b2_pre_avg + b2_rec_avg)
        else:
            f_b2 = 0.0


        log.info ("%s,%.3f,%.3f,%.3f,%.3f" % (system_name, kbp_micro_aver, b2_pre_avg, b2_rec_avg, f_b2))# b3_pre_avg, b3_rec_avg, f_b3))

        all_results["system_name"].append(float(systemOutFile.replace(".pred.tac", "").replace("t_", "")))
        all_results["kbp_micro_aver"].append(kbp_micro_aver)
        all_results["b2_pre_avg"].append(b2_pre_avg)
        all_results["b2_rec_avg"].append(b2_rec_avg)
        all_results["f_b2"].append(f_b2)
        #all_results["b3_pre_avg"].append(b3_pre_avg)
        #all_results["b3_rec_avg"].append(b3_rec_avg)
        #all_results["f_b3"].append(f_b3)

    return all_results
        #log.info "%s\t%.3f\t%.3f\t%.3f\t%.3f" % (system_name, kbp_micro_aver, b3_pre_avg, b3_rec_avg, f_b3)
 
if __name__ == "__main__":
    """
    version 0.1: initial script
    version 0.2: fixed bug in 'sameLinking' function (thanks to Timothy Nyberg) that prevented NIL elements with different identifiers on the gold std. and sys output sides from matching
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if(len(sys.argv) != 3 and len(sys.argv) != 4):
        log.info ("----------------------------------------")
        log.info ("KBP2013 Entity Linking evaluation script")
        log.info ("----------------------------------------")
        log.info ("USAGE: ./el_scorer [gold_standard_file] [system_output_dir] [focus_gold_standard_file (optional)]")
        log.info (" - gold_standard_file Ground truth for the test data.")
        log.info (" - system_output_dir  Directory with one or more system outputs for the")
        log.info ("                      test data following the KBP format.")
        log.info (" - focus_gold_standard_file same as gold_standard_file, but containing")
        log.info ("                      linkings only for queries over which query-level")
        log.info ("                      precision and recall should be averaged to compute F1.")
        log.info (" Please note that 'NIL' element identifiers must follow the format NILXXX, where XXX is a string that doesn't include white space.")
    else:
        goldStdFile = sys.argv[1]
        systemsDir = sys.argv[2]
        if (len(sys.argv) == 4):
            focusElFile = sys.argv[3]
        else:
            focusElFile = goldStdFile
        systemsRankingScript(goldStdFile, systemsDir,focusElFile)
    
