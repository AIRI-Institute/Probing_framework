# UD Filter (syntactic UD probing)
Tool that allows you to construct datasets for more complex and customizable probing tasks. It filters CoNLL-U files by syntax and morphology at the same time. 
Query language is in the form of python dictionaries (based on Grew-match).

## Usage example
1. Import the module and create an instance of the class
```python
from probing.ud_filter.filtering_probing import ProbingConlluFilter 

probing_filter = ProbingConlluFilter()
```

2. Upload files from a list of paths to files or a path to the directory.
```python
my_path = './conllu_files_dir'
probing_filter.upload_files(dir_conllu_path=my_path)
```

3. Create a query
   * simple query with one class (all unmatched sentences will be in the negative class)
   * complex query with multiple classes (only your classes will be used)
```python
# adverbal clause modifier
simple_query = {
    'ADVCL': (
        {'H': {}, 'CL': {}},
        {("H", "CL"): {"deprels": "^advcl$"}})
}

# adverbial clausal modifiers VS clausal modifiers of a noun
complex_query = {
    'ADVCL': (
        {'H': {}, 'CL': {}}, 
        {("H", "CL"): {"deprels": "^advcl$"}}),
    'ACL': (
        {'H':{}, 'CL':{},}, 
        {("H", "CL"): {"deprels": "^acl$"}})
}
```

4. Get files for your probing task
```python
probing_filter.filter_and_convert(queries=simple_query, 
                                  task_name='advcl_presence', 
                                  save_dir_path='advcl_prob_tasks')

probing_filter.filter_and_convert(queries=complex_query, 
                                  task_name='advcl_acl', 
                                  save_dir_path='advcl_acl_prob_tasks')
```

## Query guidelines

<div class="page-body"><div><table id="8060f964-e1f1-47a8-8d75-03c808e02bb8" class="simple-table"><tbody><tr id="28a5fe4e-ccc2-460e-bef7-9b06630f30a2"><td id="KPlQ" class="" style="width:232.66666666666666px"><strong>description</strong></td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><strong>grew-match</strong></td><td id="iMHy" class="" style="width:232.66666666666666px"><strong>python dictionary</strong></td></tr><tr id="0cf1cce1-b7bf-482b-a7be-3acff7d8ecda"><td id="KPlQ" class="block-color-red_background" style="width:232.66666666666666px">NODES</td><td id="&gt;w{S" class="block-color-red_background" style="width:232.66666666666666px">NODES</td><td id="iMHy" class="block-color-red_background" style="width:232.66666666666666px">NODES</td></tr><tr id="4326b0bf-991f-49e2-a097-511a02577176"><td id="KPlQ" class="" style="width:232.66666666666666px">Match<strong> any node</strong> and give it the name N</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern { N [] }</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;N&quot;: {}}</code></td></tr><tr id="16391051-3027-4a5f-9b5d-e21c7a8d0af5"><td id="KPlQ" class="" style="width:232.66666666666666px">several nodes</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern {<br>N [];<br>M [];<br>}</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;N&quot;: {}, &quot;M&quot;: {}}</code></td></tr><tr id="09c89746-9c5e-4b5e-8f33-5a3e8f67d1e6"><td id="KPlQ" class="" style="width:232.66666666666666px">Impose several <strong>restrictions</strong> on the feature structure</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern { N [ upos=VERB, Mood=Ind, Person=&quot;1&quot; ] }</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;N&quot;: {&quot;upos&quot;: &quot;VERB&quot;, &quot;mood&quot;: &quot;Ind&quot;, &quot;Person&quot;=&quot;1&quot;}} </code></td></tr><tr id="c1546e5a-afa1-4d63-9498-8259920559e5"><td id="KPlQ" class="" style="width:232.66666666666666px">Use <strong>disjunction</strong> on the feature values</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern { N [ upos=VERB, lemma=&quot;run&quot;|&quot;walk&quot;, Mood=Ind|Imp ] }</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;N&quot;: {&quot;upos&quot;: &quot;VERB&quot;, &quot;lemma&quot;: &quot;run|walk&quot;, &quot;Mood&quot;: &quot;Ind|Imp&quot;}} </code></td></tr><tr id="4e3ee281-9ce4-49ff-a126-b6203c43ba4e"><td id="KPlQ" class="" style="width:232.66666666666666px">Use <strong>negation</strong> on feature values</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern { V [ upos=VERB, VerbForm &lt;&gt; Fin|Inf ] }</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;V&quot;: {&quot;upos&quot;: &quot;VERB&quot;, &quot;VerbForm&quot;: &quot;</code><strong><code>^(?!</code></strong><code>Fin|Inf</code><strong><code>$).*$&quot;</code></strong><code>}}</code></td></tr><tr id="ec236bad-c75b-4f0b-8803-8b9aff1cfaf3"><td id="KPlQ" class="" style="width:232.66666666666666px"><strong>Regular expression</strong> can be used for feature filtering</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern {N [ form = re&quot;.*ing&quot; ]}
</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;N&quot; : {&quot;form&quot;: &quot;.*ing&quot;}}</code></td></tr><tr id="f66f9964-6bd1-4826-b70c-097bdf7aae9f"><td id="KPlQ" class="" style="width:232.66666666666666px">Require a feature to be there <strong>(without restriction on the value</strong>)</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern { N [ upos=VERB, Tense ] }</code>
</td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;N&quot;: {&quot;upos&quot;: &quot;VERB&quot;, &quot;Tense&quot;: &quot;.*&quot;}}</code></td></tr><tr id="6d70d7b6-5cfb-4919-8050-bdfc370109df"><td id="KPlQ" class="" style="width:232.66666666666666px">% Require a feature not to be there

</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern { N [ upos=VERB, !Mood ] }</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{&quot;N&quot;: {&quot;upos&quot;: &quot;^VERB$&quot;, &quot;exclude&quot;: [&quot;Mood&quot;]}}</code></td></tr><tr id="c8ef9f68-e1d7-4ab9-8ea5-e29ed05dc597"><td id="KPlQ" class="block-color-red_background" style="width:232.66666666666666px">CONSTRAINTS</td><td id="&gt;w{S" class="block-color-red_background" style="width:232.66666666666666px">CONSTRAINTS</td><td id="iMHy" class="block-color-red_background" style="width:232.66666666666666px">CONSTRAINTS</td></tr><tr id="0d8af02d-340a-417c-abe7-e25783a07b72"><td id="KPlQ" class="block-color-gray_background" style="width:232.66666666666666px">relation</td><td id="&gt;w{S" class="block-color-gray_background" style="width:232.66666666666666px">relation</td><td id="iMHy" class="block-color-gray_background" style="width:232.66666666666666px">relation</td></tr><tr id="dce64640-223b-4d72-834c-87ee98b5abe0"><td id="KPlQ" class="" style="width:232.66666666666666px">Search a <strong>relation without restriction.</strong></td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern {<br>N [];<br>M [];<br>N -&gt; M;<br>}</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{(&quot;N&quot;, &quot;M&quot;): {&quot;deprels&quot;: &quot;.*&quot;}}</code></td></tr><tr id="4b20ba7b-f456-40f2-a6fd-930492e4a719"><td id="KPlQ" class="" style="width:232.66666666666666px">Search for a <strong>given relation.</strong></td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern {<br>N -[nsubj]-&gt; M;<br>}</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{(&quot;N&quot;, &quot;M&quot;): {&quot;deprels&quot;: &quot;nsubj&quot;}}</code></td></tr><tr id="ba6cf73d-bf7c-40cd-97e3-7f27468ac7ce"><td id="KPlQ" class="block-color-gray_background" style="width:232.66666666666666px">other constraints</td><td id="&gt;w{S" class="block-color-gray_background" style="width:232.66666666666666px">other constraints</td><td id="iMHy" class="block-color-gray_background" style="width:232.66666666666666px">other constraints</td></tr><tr id="10207ffb-4983-4924-bd02-cd4c17123ddc"><td id="KPlQ" class="" style="width:232.66666666666666px">Constraint for the <strong>equality</strong> of two features</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern {<br>N -[nsubj]-&gt; M; N.Number = M.Number;<br>}</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{(&quot;N&quot;, &quot;M&quot;): {&quot;fconstraint&quot;: {&quot;intersec&quot;: [&quot;Number&quot;]}}}</code></td></tr><tr id="c43d044a-8af2-41ac-820a-dd602aca64b6"><td id="KPlQ" class="" style="width:232.66666666666666px">Constraint for the<strong> inequality</strong> of two features</td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>pattern {<br>N -[nsubj]-&gt; M; N.Number &lt;&gt; M.Number;<br>}</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{(&quot;N&quot;, &quot;M&quot;): {&quot;fconstraint&quot;: {&quot;disjoint&quot;: [&quot;Number&quot;]}}}</code></td></tr><tr id="fc9ceaa7-9713-435f-babf-72a230443e5f"><td id="KPlQ" class="" style="width:232.66666666666666px">Linear <strong>distance</strong></td><td id="&gt;w{S" class="" style="width:232.66666666666666px"><code>V &lt;&lt; S;                      % V is before S in the sentence</code></td><td id="iMHy" class="" style="width:232.66666666666666px"><code>{(&quot;V&quot;, &quot;S&quot;): {&quot;lindist&quot;: (1, inf)}}</code></td></tr></tbody></table></div><p id="2a6c8298-af14-4a86-99c7-f47741fa9a6e" class="">
</p><pre id="6e65e9a8-0756-417f-bffc-51721c9b9b2a" class="code"><code>sample_node_pattern = { node_label: {
                                field_or_category: regex_string,
                                &#x27;exclude&#x27;: [exclude categories]
                                } }
sample_constraints = { (node_label1, node_label2): {
                                &#x27;deprels&#x27;: regex_string,
                                &#x27;fconstraint&#x27;: {
                                    &#x27;disjoint&#x27;: [feature1, feature2],
                                    &#x27;intersec&#x27;: [feature1, feature2] },
                                &#x27;lindist&#x27;: (1, 10) } }</code></pre>
<p id="74ec2b99-8a39-4e43-b288-966dfa05d951" class="">Categories should be written in UD notation and style: Tense, Number, etc.</p>
</div>