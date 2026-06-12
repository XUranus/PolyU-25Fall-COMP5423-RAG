import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'RAG42 Wiki',
  tagline: 'Multi-Hop RAG System for HotpotQA',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://xuranus.github.io',
  baseUrl: '/PolyU-25Fall-COMP5423-RAG/',

  organizationName: 'XUranus',
  projectName: 'PolyU-25Fall-COMP5423-RAG',

  trailingSlash: true,

  onBrokenLinks: 'warn',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG/tree/master/wiki/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'RAG42',
      logo: {
        alt: 'RAG42 Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/docs/getting-started'},
            {label: 'Architecture', to: '/docs/architecture'},
            {label: 'Retrieval', to: '/docs/retrieval'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'GitHub', href: 'https://github.com/XUranus/PolyU-25Fall-COMP5423-RAG'},
            {label: 'HotpotQA', href: 'https://hotpotqa.github.io/'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} RAG42 Team. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'sql', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
