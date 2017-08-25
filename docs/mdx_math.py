# -*- coding: utf-8 -*-

'''
Math extension for Python-Markdown
==================================

Adds support for displaying math formulas using
[MathJax](http://www.mathjax.org/).

Author: 2015-2017, Dmitry Shachnev <mitya57@gmail.com>.
'''

"""
Copyright 2015-2017 Dmitry Shachnev <mitya57@gmail.com>.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the author nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
"""

from markdown.inlinepatterns import Pattern
from markdown.extensions import Extension
from markdown.util import AtomicString, etree


class MathExtension(Extension):
    def __init__(self, *args, **kwargs):
        self.config = {
            'enable_dollar_delimiter':
                [False, 'Enable single-dollar delimiter'],
            'add_preview': [False, 'Add a preview node before each math node'],
        }
        super(MathExtension, self).__init__(*args, **kwargs)

    def extendMarkdown(self, md, md_globals):
        def _wrap_node(node, preview_text, wrapper_tag):
            if not self.getConfig('add_preview'):
                return node
            preview = etree.Element('span', {'class': 'MathJax_Preview'})
            preview.text = AtomicString(preview_text)
            wrapper = etree.Element(wrapper_tag)
            wrapper.extend([preview, node])
            return wrapper

        def handle_match_inline(m):
            node = etree.Element('script')
            node.set('type', 'math/tex')
            node.text = AtomicString(m.group(3))
            return _wrap_node(node, ''.join(m.group(2, 3, 4)), 'span')

        def handle_match(m):
            node = etree.Element('script')
            node.set('type', 'math/tex; mode=display')
            if '\\begin' in m.group(2):
                node.text = AtomicString(''.join(m.group(2, 4, 5)))
                return _wrap_node(node, ''.join(m.group(1, 2, 4, 5, 6)), 'div')
            else:
                node.text = AtomicString(m.group(3))
                return _wrap_node(node, ''.join(m.group(2, 3, 4)), 'div')

        inlinemathpatterns = (
            Pattern(r'(?<!\\|\$)(\$)([^\$]+)(\$)'),   #  $...$
            Pattern(r'(?<!\\)(\\\()(.+?)(\\\))')      # \(...\)
        )
        mathpatterns = (
            Pattern(r'(?<!\\)(\$\$)([^\$]+)(\$\$)'),  # $$...$$
            Pattern(r'(?<!\\)(\\\[)(.+?)(\\\])'),     # \[...\]
            Pattern(r'(?<!\\)(\\begin{([a-z]+?\*?)})(.+?)(\\end{\3})')
        )
        if not self.getConfig('enable_dollar_delimiter'):
            inlinemathpatterns = inlinemathpatterns[1:]
        for i, pattern in enumerate(inlinemathpatterns):
            pattern.handleMatch = handle_match_inline
            md.inlinePatterns.add('math-inline-%d' % i, pattern, '<escape')
        for i, pattern in enumerate(mathpatterns):
            pattern.handleMatch = handle_match
            md.inlinePatterns.add('math-%d' % i, pattern, '<escape')


def makeExtension(*args, **kwargs):
    return MathExtension(*args, **kwargs)
